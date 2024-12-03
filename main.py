import math
import os
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, AutoConfig, \
    T5EncoderModel, T5Config, T5Model, T5Tokenizer, T5ForConditionalGeneration
from models.dataloader import load_data1, DataTterator2
from models.net import stage_2_features_generation1, Step_1, Step_2_forward, Step_2_reverse, Loss, Step_3_categories
from models.loss import eval_loss
from models.compute import Metric
from models.eval_features import unbatch_data
from log import logger
import time
import sys
import codecs
import random
import numpy as np
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
    """
    合并两个张量并选择概率较大的分类

    参数:
    - tensor1: 第一个张量
    - tensor2: 第二个张量

    返回值:
    - result_tensor: 合并后选择概率较大的分类的结果张量
    """
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


def set_seed(args):
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)

    # 设置随机种子
    torch.manual_seed(args.RANDOM_SEED)
    # 在使用GPU时，也要设置其随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.RANDOM_SEED)
        torch.cuda.manual_seed_all(args.RANDOM_SEED)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def train(args):
    torch.cuda.current_device()
    torch.cuda._initialized = True
    # set_seed(args)

    # torch.backends.cudnn.enabled = False

    if args.dataset == 'restaurant':
        train_path = "./datasets/Restaurant-ACOS/train.txt"
        test_path = "./datasets/Restaurant-ACOS/test.txt"
        dev_path = "./datasets/Restaurant-ACOS/dev.txt"
    elif args.dataset == 'laptop':
        train_path = "./datasets/Laptop-ACOS/train.txt"
        test_path = "./datasets/Laptop-ACOS/test.txt"
        dev_path = "./datasets/Laptop-ACOS/dev.txt"
    elif args.dataset == 'phone':
        train_path = "./datasets/Phone-ASQP/train.txt"
        test_path = "./datasets/Phone-ASQP/test.txt"
        dev_path = "./datasets/Phone-ASQP/dev.txt"
    elif args.dataset == 'laptop14':
        train_path = "./datasets/AOPE/rest16/train.txt"
        test_path = "./datasets/AOPE/rest16/test.txt"
        dev_path = "./datasets/AOPE/rest16/dev.txt"
    bert_config = AutoConfig.from_pretrained(args.init_model)
    Bert = AutoModel.from_pretrained(args.init_model, config=bert_config).get_encoder()
    # for param in Bert.embeddings.parameters():
    #     param.requires_grad = False
    # for i in range(6):
    #     for param in Bert.encoder.layer[i].parameters():
    #         param.requires_grad = False
    # bert_config = Bert.config
    Bert.to(args.device)
    # 获取bert
    bert_param_optimizer = list(Bert.named_parameters())
    step_1_model = Step_1(args, bert_config)
    step_1_model.to(args.device)
    step_1_param_optimizer = list(step_1_model.named_parameters())
    step_3_category = Step_3_categories(args)
    step_3_category.to(args.device)
    step_3_category_optimizer = list(step_3_category.named_parameters())
    step2_forward_model = Step_2_forward(args, bert_config)
    step2_forward_model.to(args.device)
    forward_step2_param_optimizer = list(step2_forward_model.named_parameters())

    step2_reverse_model = Step_2_reverse(args, bert_config)
    step2_reverse_model.to(args.device)
    reverse_step2_param_optimizer = list(step2_reverse_model.named_parameters())

    training_param_optimizer = [
        {'params': [p for n, p in bert_param_optimizer]},
        {'params': [p for n, p in step_1_param_optimizer], 'lr': args.task_learning_rate},
        {'params': [p for n, p in forward_step2_param_optimizer], 'lr': args.task_learning_rate},
        {'params': [p for n, p in reverse_step2_param_optimizer], 'lr': args.task_learning_rate},
        {'params': [p for n, p in step_3_category_optimizer], 'lr': args.task_learning_rate}
    ]
    test_datasets = load_data1(args, test_path, if_train=False)
    testset = DataTterator2(test_datasets, args)
    optimizer = AdamW(training_param_optimizer, lr=args.learning_rate, weight_decay=0.0001)
    if args.mode == 'train':
        set_seed(args)
        train_datasets = load_data1(args, train_path, if_train=False)
        trainset = DataTterator2(train_datasets, args)
        print("Train features build completed")
        print("Dev features build beginning")
        dev_datasets = load_data1(args, dev_path, if_train=False)
        devset = DataTterator2(dev_datasets, args)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        tot_loss = 0
        tot_kl_loss = 0
        bir_loss_ao, bir_loss_oa, tot_imploss_a, tot_imploss_o, tot_cat_loss = 0, 0, 0, 0, 0
        tot_con_a_loss, tot_con_o_loss = 0, 0
        best_aspect_f1, best_opinion_f1, best_APCE_f1, best_pairs_f1, best_quad_f1, best_quad_precision, best_quad_recall = 0, 0, 0, 0, 0, 0, 0
        best_quad_epoch = 0
        opinion_all_list = []
        best_pairs_precision, best_pairs_recall, best_pairs_epoch = 0, 0, 0
        num = 0
        for i in range(args.epochs):
            logger.info(('Epoch:{}'.format(i)))

            for j in tqdm.trange(trainset.batch_count):
                # set_seed(args)
                optimizer.zero_grad()
                # imp_optimizer.zero_grad()
                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                    spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                    reverse_aspect_label_tensor, related_spans_tensor, sentence_length, \
                    imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, \
                    opinion_polarity_label_tensor = trainset.get_batch(j)
                '''
                for batch_nums,opinion_value in enumerate(reverse_ner_label_tensor):
                    for index,value in enumerate(opinion_value):
                        if value.item() ==1:
                            spans_ner_label_tensor[batch_nums][index]=2
                '''
                bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)

                # aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                # cnn_spans_mask_tensor,imp_aspect_exist,imp_opinion_exist = step_1_model(bert_output.last_hidden_state,attention_mask,bert_spans_tensor,spans_mask_tensor,
                #                                                                         related_spans_tensor,bert_output.pooler_output,sentence_length,)

                aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                    cnn_spans_mask_tensor, imp_aspect_exist, imp_opinion_exist, asp_rep, opi_rep = step_1_model(
                    bert_output,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    related_spans_tensor,
                    sentence_length)

                bool_mask = spans_mask_tensor.bool()

                category_labels = [t[4][0] for t in sentence_length]
                pairs = [t[3] for t in sentence_length if len(t[3]) > 0]
                if len(pairs) > 0:
                    category_logits, category_label = step_3_category(spans_embedding, bert_spans_tensor, pairs,
                                                                      spans_mask_tensor, category_labels)
                is_aspect = True
                category_label = category_label.to(torch.long).to(args.device)
                '''Batch更新'''
                # 21X420                     21 1 768           21 100 768           21 100        21 420 768    21 420
                # 从输入的 BERT 特征、注意力掩码、span 序列和 span 掩码中生成经过处理后的特征
                if len(spans_aspect_tensor) > 0:
                    all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                        all_spans_embedding, step2_forward_embedding, all_span_mask, all_left_forward_tensor, all_right_forward_tensor = stage_2_features_generation1(
                        bert_output.last_hidden_state,
                        attention_mask, bert_spans_tensor,
                        spans_mask_tensor, forward_embedding, reverse_embedding,
                        spans_aspect_tensor,
                        is_aspect, asp_rep,
                        spans_opinion_label_tensor,
                        )
                    step_2_opinion_class_logits = \
                        step2_forward_model(all_spans_embedding, all_span_mask, all_span_aspect_tensor,
                                            step2_forward_embedding)
                is_aspect = False
                if len(reverse_opinion_tensor) > 0:
                    all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                        reverse_spans_embedding, step2_reverse_embedding, reverse_span_mask, all_left_reverse_tensor, all_right_reverse_tensor = stage_2_features_generation1(
                        bert_output.last_hidden_state,
                        attention_mask, bert_spans_tensor,
                        spans_mask_tensor, reverse_embedding,
                        forward_embedding,
                        reverse_opinion_tensor,
                        is_aspect, opi_rep,
                        reverse_aspect_label_tensor,
                        )
                    step_2_aspect_class_logits = step2_reverse_model(reverse_spans_embedding,
                                                                     reverse_span_mask, all_reverse_opinion_tensor,
                                                                     step2_reverse_embedding)
                # 计算loss和KL loss
                loss, kl_loss, as_2_op_loss, op_2_as_loss, imp_aspect_loss, imp_opinion_loss, category_loss, con_asp_loss, con_opi_loss = \
                    Loss(spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor,
                         step_2_opinion_class_logits, spans_mask_tensor, all_span_mask,
                         reverse_ner_label_tensor, opinion_class_logits, all_reverse_aspect_tensor,
                         step_2_aspect_class_logits,
                         cnn_spans_mask_tensor, reverse_span_mask, spans_embedding, related_spans_tensor,
                         category_label, category_logits,
                         imp_asp_label_tensor, imp_opi_label_tensor, imp_aspect_exist, imp_opinion_exist,
                         sentence_length, asp_rep, opi_rep, len(spans_aspect_tensor), args)

                if args.accumulation_steps > 1:
                    # loss = loss / args.accumulation_steps
                    loss.backward()
                    if ((j + 1) % args.accumulation_steps) == 0 or (j + 1) == trainset.batch_count:
                        # nn.utils.clip_grad_norm_(list(step_1_model.parameters())+list(step2_forward_model.parameters())+
                        #                          list(step2_reverse_model.parameters())+list(step_3_category.parameters()), max_norm=1, norm_type=2)
                        optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()

                tot_loss += loss.item()
                tot_kl_loss += kl_loss
                bir_loss_ao += as_2_op_loss.item()
                bir_loss_oa += op_2_as_loss.item()
                tot_imploss_a += imp_aspect_loss.item()
                tot_imploss_o += imp_opinion_loss.item()
                tot_cat_loss += category_loss.item()
                tot_con_a_loss += con_asp_loss.item()
                tot_con_o_loss += con_opi_loss.item()
            logger.info(('Loss:', tot_loss))
            logger.info(('KL_Loss:', tot_kl_loss))
            logger.info(('ao_Loss:', bir_loss_ao))
            logger.info(('oa_Loss:', bir_loss_oa))
            logger.info(('imp_aspect_loss:', tot_imploss_a))
            logger.info(('imp_opinion_loss:', tot_imploss_o))
            logger.info(('category_loss:', tot_cat_loss))
            logger.info(('con_aspect_loss:', tot_con_a_loss))
            logger.info(('con_opinion_loss:', tot_con_o_loss))

            tot_loss = 0
            tot_kl_loss = 0

            bir_loss_ao, bir_loss_oa, tot_imploss_o, tot_imploss_a, tot_cat_loss = 0, 0, 0, 0, 0
            tot_con_a_loss, tot_con_o_loss = 0, 0

            quad_result, opinion_imp_list, pair_result = eval(Bert, step_1_model, step2_forward_model,
                                                              step2_reverse_model, step_3_category, devset, args)
            if opinion_imp_list != []:
                opinion_all_list.append(opinion_imp_list)
            if pair_result[2] > best_pairs_f1:
                best_pairs_f1 = pair_result[2]
                best_pairs_precision = pair_result[0]
                best_pairs_recall = pair_result[1]
                best_pairs_epoch = i
            if len(opinion_all_list) > 10:
                common_sentences = set(opinion_all_list[0])
                # with open("common_strings" + str(num) + ".txt", "w") as file:
                #     for string in common_sentences:
                #         file.write(string + "\n")
                num += 1
                opinion_all_list = []
            if quad_result[2] > best_quad_f1:
                if args.dataset == "laptop":
                    limit = 0.42
                elif args.dataset == "restaurant":
                    limit = 0.62
                elif args.dataset == "phone":
                    limit = 0.55
                else:
                    limit = 1
                if quad_result[2] > limit:
                    model_path = args.model_dir + args.dataset + '_' + str(quad_result[2]) + '.pt'
                    state = {
                        "bert_model": Bert.state_dict(),
                        "step_1_model": step_1_model.state_dict(),
                        "step2_forward_model": step2_forward_model.state_dict(),
                        "step2_reverse_model": step2_reverse_model.state_dict(),
                        "step_3_category": step_3_category.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    torch.save(state, model_path)
                    logger.info("_________________________________________________________")
                    logger.info("best model save")
                    logger.info("_________________________________________________________")

                    best_quad_f1 = quad_result[2]
                    best_quad_precision = quad_result[0]
                    best_quad_recall = quad_result[1]
                    best_quad_epoch = i

            logger.info(
                'best quad epoch: {}\tbest quad precision: {:.8f}\tbest quad recall: {:.8f}\tbest quad f1: {:.8f}'.
                format(best_quad_epoch, best_quad_precision, best_quad_recall, best_quad_f1))
            logger.info(
                'best pair epoch: {}\tbest pair precision: {:.8f}\tbest pair recall: {:.8f}\tbest pair f1: {:.8f}'.
                format(best_pairs_epoch, best_pairs_precision, best_pairs_recall, best_pairs_f1))
        logger.info(
            'best quad epoch: {}\tbest quad precision: {:.8f}\tbest quad recall: {:.8f}\tbest quad f1: {:.8f}'.
            format(best_quad_epoch, best_quad_precision, best_quad_recall, best_quad_f1))
        logger.info(
            'best pair epoch: {}\tbest pair precision: {:.8f}\tbest pair recall: {:.8f}\tbest pair f1: {:.8f}'.
            format(best_pairs_epoch, best_pairs_precision, best_pairs_recall, best_pairs_f1))

    logger.info("Features build completed")
    logger.info("Evaluation on testset:")

    model_path = args.model_dir + args.dataset + '_' + str(best_quad_f1) + '.pt'

    if args.muti_gpu:
        state = torch.load(model_path)
    else:
        state = torch.load(model_path)
        # state = load_with_single_gpu(model_path)

    Bert.load_state_dict(state['bert_model'])
    step_1_model.load_state_dict(state['step_1_model'])
    step2_forward_model.load_state_dict(state['step2_forward_model'])
    step2_reverse_model.load_state_dict(state['step2_reverse_model'])
    step_3_category.load_state_dict(state['step_3_category'])
    bert_params = sum(p.numel() for p in Bert.parameters())
    step_1_model_params = sum(p.numel() for p in step_1_model.parameters())
    step2_forward_model_params = sum(p.numel() for p in step2_forward_model.parameters())
    step2_reverse_model_params = sum(p.numel() for p in step2_reverse_model.parameters())
    step_3_category_params = sum(p.numel() for p in step_3_category.parameters())
    total_params = bert_params + step_1_model_params + step2_forward_model_params + step2_reverse_model_params + step_3_category_params
    print("the number of all model:", total_params)
    hyper_result, opinion_imp_list, pair_result = eval(Bert, step_1_model, step2_forward_model, step2_reverse_model,
                                                       step_3_category, testset, args)
    return hyper_result
