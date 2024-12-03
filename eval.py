import torch
import torch.nn.functional as F
from models.net import stage_2_features_generation1, Step_1, Step_2_forward, Step_2_reverse, Loss, Step_3_categories, \
    eval_loss
from models.compute import Metric
from models.eval_features import unbatch_data
from log import logger
import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sentiment2id = {'none': 3, 'positive': 2, 'negative': 0, 'neutral': 1}
def cartesian_product(tensor_list1, tensor_list2):
    # 创建两个字典，分别用于按 batch_num 分组
    grouped_tensors1 = {}
    grouped_tensors2 = {}

    # 将 tensor_list1 中的 tensor 按第一个元素 batch_num 分组
    for tensor in tensor_list1:
        batch_num = int(tensor[0][0])
        if batch_num in grouped_tensors1:
            grouped_tensors1[batch_num].append(tensor[0][1:])
        else:
            grouped_tensors1[batch_num] = [tensor[0][1:]]

    # 将 tensor_list2 中的 tensor 按第一个元素 batch_num 分组
    for tensor in tensor_list2:
        batch_num = int(tensor[0][0])
        if batch_num in grouped_tensors2:
            grouped_tensors2[batch_num].append(tensor[0][1:])
        else:
            grouped_tensors2[batch_num] = [tensor[0][1:]]

    # 创建一个列表，用于存储结果
    result = []

    # 找到两个 tensor 列表共同拥有的 batch_num
    common_batch_nums = set(grouped_tensors1.keys()) & set(grouped_tensors2.keys())

    # 对每个共同的 batch_num 进行处理
    for batch_num in common_batch_nums:
        tensors1 = grouped_tensors1[batch_num]
        tensors2 = grouped_tensors2[batch_num]

        # 将 tensors1 和 tensors2 移动到相同的设备上
        device = tensors1[0].device if len(tensors1) > 0 else tensors2[0].device
        tensors1 = [tensor.to(device) for tensor in tensors1]
        tensors2 = [tensor.to(device) for tensor in tensors2]

        # 对 tensors1 和 tensors2 进行笛卡尔积拼接
        for tensor1 in tensors1:
            for tensor2 in tensors2:
                concatenated_tensor = torch.cat([tensor1, tensor2, torch.tensor([batch_num], device=device)])
                result.append(concatenated_tensor)

    return result


def merge_and_choose_max(tensor1, tensor2):
    greater_than_tensor = tensor1 > tensor2

    # 创建一个空的结果张量
    result_tensor = torch.zeros(tensor1.size(0), dtype=torch.float32)

    # 对每一行进行处理，选择概率较大的分类
    for i in range(tensor1.size(0)):
        common_max = torch.argmax(tensor1[i] * greater_than_tensor[i] + tensor2[i] * ~greater_than_tensor[i])
        result_tensor[i] = common_max

    return result_tensor


def cal_step1(gold_aspect_label, pred_aspect_label, spans_mask_tensor, total_tp, total_fp, total_fn):
    gold_aspect_label = gold_aspect_label.cpu().numpy()  # 将张量转换为NumPy数组
    pred_aspect_label = pred_aspect_label.cpu().numpy()  # 将张量转换为NumPy数组
    for index, (true_sample, predicted_sample) in enumerate(zip(gold_aspect_label, pred_aspect_label)):
        last_index = torch.sum(spans_mask_tensor[index]).item() - 1
        for label_index, (true_label, predicted_label) in enumerate(zip(true_sample, predicted_sample)):
            if label_index == 0 or label_index == last_index:
                continue
            if true_label == 1 and predicted_label == 1:
                total_tp += 1
            elif true_label == 0 and predicted_label == 1:
                total_fp += 1
            elif true_label == 1 and predicted_label == 0:
                total_fn += 1
            else:
                continue
    return total_tp, total_fp, total_fn


def cal_f1(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def cal_softmax(step_2_logits, all_span_mask):
    bool_mask = all_span_mask.bool()
    masked_logits = step_2_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
    soft = F.softmax(masked_logits, dim=1)[:, :, 1:]
    final_soft = torch.sum(soft, dim=-1) / len(soft)
    return final_soft


def cal_imp_num(asp_0, asp_1, asp_2, right_a0, right_a1, right_a2, index, label_tensor, pred_list):
    asp_0 += len(label_tensor[label_tensor == 0])
    asp_1 += len(label_tensor[label_tensor == 1])
    asp_2 += len(label_tensor[label_tensor == 2])
    for i, value in enumerate(label_tensor):
        if value == 0:
            if pred_list[i] == 0:
                right_a0 += 1
        elif value == 1:
            if pred_list[i] == 1:
                right_a1 += 1
        else:
            if pred_list[i] == 2:
                right_a2 += 1
    return asp_0, asp_1, asp_2, right_a0, right_a1, right_a2


def eval(bert_model, step_1_model, step_2_forward, step_2_reverse, step_3_category, dataset, args):
    with torch.no_grad():
        bert_model.eval()
        step_1_model.eval()
        step_2_forward.eval()
        step_2_reverse.eval()
        step_3_category.eval()
        '''真实结果'''
        gold_instances = []
        tot_aspcet_loss = 0
        tot_opinion_loss = 0
        tot_imp_asp_loss = 0
        tot_imp_opi_loss = 0

        '''前向预测结果'''
        forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment, \
            forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result, \
            forward_stage2_pred_opinion_sentiment_logit, forward_pred_category_logit = [], [], [], [], [], []
        category_result = []
        '''反向预测结果'''
        reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment, \
            reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result, \
            reverse_stage2_pred_aspect_sentiment_logit = [], [], [], [], []

        '''隐式通道预测结果'''
        aspect_imp_opinion_result, opinion_imp_aspect_result = [], []

        opinion_imp_list = []
        aspect_tp = 0
        aspect_fp = 0
        aspect_fn = 0

        opinion_tp = 0
        opinion_fp = 0
        opinion_fn = 0
        asp_0, asp_1, asp_2, right_a0, right_a1, right_a2 = 0, 0, 0, 0, 0, 0
        opi_0, opi_1, opi_2, right_o0, right_o1, right_o2 = 0, 0, 0, 0, 0, 0
        tensor_2d_a = None
        tensor_2d_o = None
        tensor_2d_a_label = None
        tensor_2d_o_label = None
        origin_rep = None
        # set_seed(args)
        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length, \
                imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, \
                opinion_polarity_label_tensor = dataset.get_batch(j)

            bert_features = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                cnn_spans_mask_tensor, imp_aspect_exist, imp_opinion_exist, asp_rep, opi_rep = step_1_model(
                bert_features, attention_mask, bert_spans_tensor, spans_mask_tensor,
                related_spans_tensor, sentence_length)

            '''Batch更新'''
            pred_aspect_logits = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits = torch.where(spans_mask_tensor == 1, pred_aspect_logits,
                                             torch.tensor(0).type_as(pred_aspect_logits))

            pred_imp_aspect = torch.argmax(F.softmax(imp_aspect_exist, dim=1), dim=1)
            pred_imp_opinion = torch.argmax(F.softmax(imp_opinion_exist, dim=1), dim=1)
            asp_0, asp_1, asp_2, right_a0, right_a1, right_a2 = cal_imp_num(asp_0, asp_1, asp_2, right_a0, right_a1,
                                                                            right_a2, 0, imp_asp_label_tensor,
                                                                            pred_imp_aspect)
            opi_0, opi_1, opi_2, right_o0, right_o1, right_o2 = cal_imp_num(opi_0, opi_1, opi_2, right_o0, right_o1,
                                                                            right_o2, 1, imp_opi_label_tensor,
                                                                            pred_imp_opinion)
            # pred_imp_aspect[pred_imp_aspect>1]=1
            # pred_imp_opinion[pred_imp_opinion>1]=1
            # pred_imp_aspect = imp_asp_label_tensor
            # pred_imp_opinion = imp_opi_label_tensor
            # origin = bert_features.pooler_output.clone().cpu()
            origin = spans_embedding[:, 0, :].clone().cpu()
            if tensor_2d_a is None:
                tensor_2d_a = asp_rep
                tensor_2d_o = opi_rep
                tensor_2d_a_label = imp_asp_label_tensor
                tensor_2d_o_label = imp_opi_label_tensor
                origin_rep = origin
            else:
                tensor_2d_a = torch.cat((tensor_2d_a, asp_rep), dim=0)
                tensor_2d_o = torch.cat((tensor_2d_o, opi_rep), dim=0)
                tensor_2d_a_label = torch.cat((tensor_2d_a_label, imp_asp_label_tensor), dim=0)
                tensor_2d_o_label = torch.cat((tensor_2d_o_label, imp_opi_label_tensor), dim=0)
                origin_rep = torch.cat((origin_rep, origin), dim=0)
            # tensor_2d_o = TSNE(n_components=2).fit_transform(opi_rep.cpu().numpy())
            # writer.add_embedding(tensor_2d_a, metadata=imp_asp_label_tensor, tag='tensor2')
            reverse_pred_stage1_logits = torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2)
            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)
            reverse_pred_stage1_logits = torch.where(spans_mask_tensor == 1, reverse_pred_stage1_logits,
                                                     torch.tensor(0).type_as(reverse_pred_stage1_logits))

            aspect_tp, aspect_fp, aspect_fn = cal_step1(spans_ner_label_tensor, pred_aspect_logits, spans_mask_tensor,
                                                        aspect_tp, aspect_fp, aspect_fn)
            opinion_tp, opinion_fp, opinion_fn = cal_step1(reverse_ner_label_tensor, reverse_pred_stage1_logits,
                                                           spans_mask_tensor, opinion_tp, opinion_fp, opinion_fn)
            for i in range(len(pred_imp_aspect)):
                pred_aspect_logits[i][0] = pred_imp_aspect[i]
                pred_aspect_logits[i][torch.sum(spans_mask_tensor[i]).item() - 1] = 0
                reverse_pred_stage1_logits[i][torch.sum(spans_mask_tensor[i]).item() - 1] = pred_imp_opinion[i]
                reverse_pred_stage1_logits[i][0] = 0
            '''真实结果合成'''
            gold_instances.append(dataset.instances[j])
            result = []

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] != 0 \
                    and torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] != 0:
                opinion_span = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                           torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                aspect_span = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                          torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                # pred_pair = torch.tensor(
                #     [[aspect_span[i][0][1], opinion_span[j][0][1]]
                #      for i in range(len(aspect_span)) for j in range(len(opinion_span))])
                pred_pair = cartesian_product(aspect_span, opinion_span)
                if len(pred_pair) == 0:
                    category_result.append([])
                # aspect_rep = []
                # opinion_rep = []
                # for aspect_idx in aspect_span:
                #     aspect_rep.append(spans_embedding[aspect_idx[0][0], aspect_idx[0][1]])
                # aspect_rep = torch.stack(aspect_rep)
                # for opinion_idx in opinion_span:
                #     opinion_rep.append(spans_embedding[opinion_idx[0][0], opinion_idx[0][1]])
                # opinion_rep = torch.stack(opinion_rep)
                else:
                    input_rep = []
                    for span in pred_pair:
                        batch = int(span[2])
                        aspect_index = int(span[0])
                        opinion_index = int(span[1])
                        # aspect_rep = forward_embedding[batch,aspect_index]
                        # opinion_rep = reverse_embedding[batch, opinion_index]
                        aspect_rep = spans_embedding[batch, aspect_index]
                        opinion_rep = spans_embedding[batch, opinion_index]
                        last_index = torch.sum(spans_mask_tensor[batch]).item() - 1
                        # if aspect_index == 0 and opinion_index != last_index:
                        #     left = torch.mean(spans_embedding[batch, 1:opinion_index, :],
                        #                         dim=0) if opinion_index != 1 else \
                        #         spans_embedding[batch][0]
                        #     right= torch.mean(spans_embedding[batch, opinion_index:last_index, :],
                        #                          dim=0) if opinion_index != last_index else \
                        #         spans_embedding[batch][last_index]
                        # elif aspect_index != 0 and opinion_index == last_index:
                        #     left = torch.mean(spans_embedding[batch, 1:aspect_index, :],
                        #                         dim=0) if aspect_index != 1 else \
                        #         spans_embedding[batch][0]
                        #     right = torch.mean(spans_embedding[batch, aspect_index:last_index, :],
                        #                          dim=0) if aspect_index != last_index else \
                        #         spans_embedding[batch][last_index]
                        # elif aspect_index == 0 and opinion_index == last_index:
                        #     left = torch.mean(spans_embedding[batch, 1:last_index, :], dim=0)
                        #     right = torch.mean(spans_embedding[batch, 1:last_index, :], dim=0)
                        # else:
                        #     left = torch.mean(spans_embedding[batch, 1:aspect_index, :],
                        #                         dim=0) if aspect_index != 1 else \
                        #         spans_embedding[batch][0]
                        #     right = torch.mean(spans_embedding[batch, aspect_index:last_index, :],
                        #                          dim=0) if aspect_index != last_index else \
                        #         spans_embedding[batch][last_index]
                        # input_rep.append(torch.cat((aspect_rep + opinion_rep, left, right), dim=0))
                        input_rep.append(torch.cat((aspect_rep, opinion_rep), dim=0))
                    input_rep = torch.stack(input_rep)
                    # # 使用广播计算笛卡尔积并拼接
                    # expanded_aspect = aspect_rep.unsqueeze(1)
                    # expanded_opinion = opinion_rep.unsqueeze(0)
                    # expanded_aspect = expanded_aspect.repeat(1, opinion_rep.size(0), 1)
                    # expanded_opinion = expanded_opinion.repeat(aspect_rep.size(0), 1, 1)
                    # cartesian_product1 = torch.cat((expanded_aspect, expanded_opinion), dim=2)
                    # # 转换形状
                    # input_rep  = cartesian_product1.reshape(aspect_rep.size(0) * opinion_rep.size(0), 768 * 2)
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, input_rep,
                                                         spans_mask_tensor)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    result = []
                    for i, pair in enumerate(pred_pair):
                        a_o_c_result = [int(pair[0]), int(pair[1]), int(pred_category_logits[i]), int(pair[2])]
                        result.append(a_o_c_result)
                    category_result.append(result)
            else:
                category_result.append([])

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:

                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))

                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)

                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                if len(spans_opinion_label_tensor) != 0:
                    forward_stage2_pred_opinion_sentiment_logit.append(
                        torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
                else:
                    forward_stage2_pred_opinion_sentiment_logit.append(torch.tensor([]))
                # forward_pred_category_logit.append(pred_category_logits)
                aspect_imp_opinion_result.append([])

            else:
                pred_aspect_spans = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                                torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                pred_span_aspect_tensor = None
                for pred_aspect_span in pred_aspect_spans:
                    batch_num = pred_aspect_span.squeeze()[0]
                    # if int(pred_aspect_span.squeeze()[1]) == len(bert_spans_tensor[batch_num]) - 1:
                    #     continue
                    aspect_index = pred_aspect_span.squeeze()[1]
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1],
                         aspect_index)).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt),
                                                            dim=0)

                is_aspect = True
                _, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, step2_forward_embedding, all_span_mask, \
                    all_left_tensor, all_right_tensor = stage_2_features_generation1(
                    bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    forward_embedding, reverse_embedding, pred_span_aspect_tensor, is_aspect, asp_rep)

                opinion_class_logits1 = step_2_forward(all_spans_embedding, all_span_mask,
                                                       all_span_aspect_tensor, step2_forward_embedding)
                # step1. 找到所有的opinion_index及其表征
                # step2. 遍历aspect个数，逐个拼接opinion表征，最后输入到step3_category中
                # step3. 添加到result

                aspect_imp_result = []
                aspect_imp_opinion_result.append(aspect_imp_result)

                forward_stage1_pred_aspect_result.append(pred_span_aspect_tensor)
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.argmax(F.softmax(opinion_class_logits1, dim=2), dim=2))
                forward_stage2_pred_opinion_sentiment_logit.append(F.softmax(opinion_class_logits1, dim=2))
                # forward_pred_category_logit.append(pred_category_logits)
            '''反向预测'''
            if torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] == 0:
                reverse_stage1_pred_opinion_result.append(torch.full_like(reverse_opinion_tensor, -1))
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.full_like(reverse_aspect_label_tensor, -1))
                if len(reverse_aspect_label_tensor) != 0:
                    reverse_stage2_pred_aspect_sentiment_logit.append(
                        torch.full_like(reverse_aspect_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)),
                                        -1))
                else:
                    reverse_stage2_pred_aspect_sentiment_logit.append(torch.tensor([]))
                opinion_imp_aspect_result.append([])
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                         torch.nonzero(reverse_pred_stage1_logits,
                                                                       as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                    # if int(reverse_pred_opinion_span.squeeze()[1]) == 0:
                    #    continue
                    opinion_index = reverse_pred_opinion_span.squeeze()[1]
                    reverse_opinion_tensor_unspilt = bert_spans_tensor[batch_num,
                                                     reverse_pred_opinion_span.squeeze()[1], :2]
                    reverse_opinion_tensor_unspilt = torch.tensor(
                        (batch_num, reverse_opinion_tensor_unspilt[0], reverse_opinion_tensor_unspilt[1],
                         opinion_index)).unsqueeze(0)
                    if reverse_span_opinion_tensor is None:
                        reverse_span_opinion_tensor = reverse_opinion_tensor_unspilt
                    else:
                        reverse_span_opinion_tensor = torch.cat(
                            (reverse_span_opinion_tensor, reverse_opinion_tensor_unspilt), dim=0)
                is_aspect = False
                __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                    reverse_spans_embedding, step2_reverse_embedding, reverse_span_mask, all_left_tensor, all_right_tensor = stage_2_features_generation1(
                    bert_features.last_hidden_state,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    reverse_embedding,
                    forward_embedding,
                    reverse_span_opinion_tensor,
                    is_aspect,
                    opi_rep
                )

                reverse_aspect_class_logits = step_2_reverse(reverse_spans_embedding, reverse_span_mask,
                                                             all_reverse_opinion_tensor, step2_reverse_embedding)

                opinion_imp_result = []

                opinion_imp_aspect_result.append(opinion_imp_result)

                output_values, output_indices = torch.max(reverse_aspect_class_logits, dim=-1)
                aspect_list = []
                for row in output_indices:
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    aspect_list.append(non_three_indices_row)

                reverse_stage1_pred_opinion_result.append(reverse_span_opinion_tensor)
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(
                    torch.argmax(F.softmax(reverse_aspect_class_logits, dim=2), dim=2))
                reverse_stage2_pred_aspect_sentiment_logit.append(F.softmax(reverse_aspect_class_logits, dim=2))

            exist_aspect = spans_ner_label_tensor[:, 0]
            exist_opinion = reverse_ner_label_tensor[
                range(reverse_ner_label_tensor.shape[0]), torch.sum(spans_mask_tensor, dim=-1) - 1]

            # category_result.append(result)
            aspect_loss, opinion_loss, imp_asp_loss, imp_opi_loss = \
                eval_loss(spans_ner_label_tensor, aspect_class_logits, reverse_ner_label_tensor, opinion_class_logits,
                          spans_mask_tensor, exist_aspect, exist_opinion, imp_aspect_exist,
                          imp_opinion_exist, args)

            tot_aspcet_loss += aspect_loss.item()
            tot_opinion_loss += opinion_loss.item()
            tot_imp_asp_loss += imp_asp_loss.item()
            tot_imp_opi_loss += imp_opi_loss.item()
        tensor_2d_a = tensor_2d_a.cpu().numpy()
        tensor_2d_o = tensor_2d_o.cpu().numpy()
        tensor_2d_a_label = tensor_2d_a_label.cpu().numpy()
        tensor_2d_o_label = tensor_2d_o_label.cpu().numpy()
        origin_rep = origin_rep.cpu().numpy()
        tensor_2d_asp = TSNE(n_components=2, random_state=2024).fit_transform(origin_rep)
        class0 = tensor_2d_asp[tensor_2d_a_label == 0]
        class1 = tensor_2d_asp[tensor_2d_a_label == 1]

        plt.scatter(class0[:, 0], class0[:, 1], label='without implicit aspect', c='lightgreen', edgecolors='black',
                    linewidth=1)
        plt.scatter(class1[:, 0], class1[:, 1], label='with implicit aspect', c='cornflowerblue', edgecolors='black',
                    linewidth=1)
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        plt.title('Representation Space without the contrastive learning')
        plt.legend()
        plt.show()

        gold_instances = [x for i in gold_instances for x in i]
        forward_pred_data = (forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment,
                             forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result,
                             forward_stage2_pred_opinion_sentiment_logit, aspect_imp_opinion_result, category_result)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit, opinion_imp_aspect_result)
        reverse_pred_result = unbatch_data(reverse_pred_data)

        metric = Metric(args, forward_pred_result, reverse_pred_result, gold_instances)
        quad_result, pair_result, low_standard, triples_result = metric.score_triples()

        aspect_p, aspect_r, aspect_f = cal_f1(aspect_tp, aspect_fp, aspect_fn)
        opinion_p, opinion_r, opinion_f = cal_f1(opinion_tp, opinion_fp, opinion_fn)
        logger.info('total_aspect_loss:{}'.format(tot_aspcet_loss))
        logger.info('total_opinion_loss:{}'.format(tot_opinion_loss))
        logger.info('total_imp_aspect_loss:{}'.format(tot_imp_asp_loss))
        logger.info('total_imp_opinion_loss:{}'.format(tot_imp_opi_loss))

        logger.info('step1 predict aspect,precision:{},recall:{}，F1:{}'.format(aspect_p, aspect_r, aspect_f))
        logger.info('step1 predict opinion,precision:{},recall:{}，F1:{}'.format(opinion_p, opinion_r, opinion_f))
        # logger.info('step1 all aspect({}/{})correct:{:.8f}, all opinion({}/{})correct:{:.8f}'.format(pred_imp_right_aspect,pred_imp_aspect_total,
        #                                                                                         pred_imp_right_aspect/pred_imp_aspect_total,
        #                                                                             pred_imp_right_opinion, pred_imp_opinion_total,
        #                                                                             pred_imp_right_opinion/pred_imp_opinion_total))
        logger.info(
            'explicit aspect({}/{}),implicit aspect:({}/{}), mixed aspect({}/{}),'.format(right_a0, asp_0, right_a1,
                                                                                          asp_1, right_a2, asp_2))
        logger.info(
            'explicit opinion({}/{}),implicit opinion:({}/{}), mixed opinion({}/{}),'.format(right_o0, opi_0, right_o1,
                                                                                             opi_1, right_o2, opi_2))

        # logger.info('single impilict aspect({}/{})correct:{:.8f},single impilict opinion({}/{})correct:{:.8f},，all implicit({}/{})correct {:.8f}:explicit is:{}'.format(pred_imp_aspect_num,pred_imp_aspect_num_total,
        #     pred_imp_aspect_num/ pred_imp_aspect_num_total,pred_imp_opinion_num,pred_imp_opinion_num_total,
        #     pred_imp_opinion_num / pred_imp_opinion_num_total,pred_imp_asp_opinion_num,pred_imp_asp_opinion_num_total,
        #                 pred_imp_asp_opinion_num/pred_imp_asp_opinion_num_total,pred_emp_num_total))
        # if pred_imp_opinion_num / pred_imp_opinion_num_total < 0.60:
        #     opinion_imp_list = []

        logger.info('pair precision: {}\tpair recall: {:.8f}\tpair f1: {:.8f}'.format(pair_result[0],
                                                                                      pair_result[1],
                                                                                      pair_result[2]))
        logger.info('quad precision: {}\tquad recall: {:.8f}\tquad f1: {:.8f}'.format(quad_result[0],
                                                                                      quad_result[1],
                                                                                      quad_result[2]))
        logger.info('triples precision: {}\ttriples recall: {:.8f}\ttriples f1: {:.8f}'.format(triples_result[0],
                                                                                               triples_result[1],
                                                                                               triples_result[2]))

    bert_model.train()
    step_1_model.train()
    step_2_forward.train()
    step_2_reverse.train()
    step_3_category.train()
    return quad_result, opinion_imp_list, pair_result