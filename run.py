import math
import os
import argparse
import tqdm
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from models.data_BIO_loader import load_data, load_data1,DataTterator,DataTterator2
from models.model import stage_2_features_generation, stage_2_features_generation1,Step_1, Step_2_forward, Step_2_reverse, Loss,Step_3_categories
from models.Metric import Metric
from models.eval_features import unbatch_data
from log import logger
from thop import profile, clever_format
import time
import sys
import codecs
import GPUtil
from itertools import groupby
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
                concatenated_tensor = torch.cat([tensor1, tensor2,torch.tensor([batch_num], device=device)])
                result.append(concatenated_tensor)

    return result

def cal_step1(gold_aspect_label,pred_aspect_label,spans_mask_tensor,total_tp,total_fp,total_fn):
    gold_aspect_label = gold_aspect_label.cpu().numpy()  # 将张量转换为NumPy数组
    pred_aspect_label = pred_aspect_label.cpu().numpy()  # 将张量转换为NumPy数组
    for index,(true_sample, predicted_sample) in enumerate(zip(gold_aspect_label, pred_aspect_label)):
        last_index = torch.sum(spans_mask_tensor[index]).item()-1
        for label_index,(true_label, predicted_label) in enumerate(zip(true_sample, predicted_sample)):
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
    return total_tp,total_fp,total_fn

def cal_f1(total_tp,total_fp,total_fn):
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision,recall,f1
def eval(bert_model, step_1_model, step_2_forward, step_2_reverse,step_3_category, dataset, args):
    with torch.no_grad():
        bert_model.eval()
        step_1_model.eval()
        step_2_forward.eval()
        step_2_reverse.eval()
        step_3_category.eval()
        '''真实结果'''
        gold_instances = []
        '''前向预测结果'''
        forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment, \
        forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result, \
        forward_stage2_pred_opinion_sentiment_logit,forward_pred_category_logit = [],[],[],[],[],[]
        category_result = []
        '''反向预测结果'''
        reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment, \
        reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result, \
        reverse_stage2_pred_aspect_sentiment_logit = [], [], [], [], []

        '''隐式通道预测结果'''
        aspect_imp_opinion_result,opinion_imp_aspect_result = [],[]
        is_aspect = False
        pred_imp_right_aspect = 0
        pred_imp_right_opinion = 0
        pred_imp_aspect_total = 0
        pred_imp_opinion_total = 0

        pred_imp_aspect_num = 0
        pred_imp_opinion_num = 0
        pred_imp_asp_opinion_num = 0
        pred_imp_aspect_num_total = 0
        pred_imp_opinion_num_total = 0
        pred_imp_asp_opinion_num_total = 0
        pred_emp_num_total = 0
        opinion_imp_list = []
        aspect_tp = 0
        aspect_fp = 0
        aspect_fn = 0

        opinion_tp = 0
        opinion_fp = 0
        opinion_fn = 0
        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
            reverse_aspect_label_tensor, related_spans_tensor, sentence_length, \
            imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, \
            opinion_polarity_label_tensor = dataset.get_batch(j)

            bert_features = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)

            # if j == 0 and args.model_para_test:
            #     step_1_model.to("cpu")
            #
            #     flop_step1, para_step1 = profile(step_1_model, inputs=(bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
            #                                                          related_spans_tensor, sentence_length))
            #     macs, param = clever_format([flop_step1,para_step1], "%.3f")
            #     print("STEP 1 MACs: ", macs, "STEP 1 Params", param)
            #     logger.info(
            #         'STEP 1 MACs:  {}\tSTEP 1 Params: {:.8f}\n\n'.format(macs, param))

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
            cnn_spans_mask_tensor,imp_aspect_exist,imp_opinion_exist = step_1_model(
                bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                related_spans_tensor, bert_features.pooler_output,sentence_length)



            '''Batch更新'''
            pred_aspect_logits = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits = torch.where(spans_mask_tensor == 1, pred_aspect_logits,
                                             torch.tensor(0).type_as(pred_aspect_logits))


            pred_imp_aspect = torch.argmax(F.softmax(imp_aspect_exist,dim=1),dim=1)
            pred_imp_opinion = torch.argmax(F.softmax(imp_opinion_exist,dim=1),dim=1)

            # pred_category_logits = torch.where(spans_mask_tensor == 1,pred_category_logits,
            #                                    torch.tensor(0).type_as(pred_category_logits))

            for i, value in enumerate(sentence_length):
                flag1 = 0
                flag2 = 0
                if int(pred_imp_aspect[i]) == 1:
                    for element in value[3]:
                        if element[0] == [0, 0, 0]:
                            pred_imp_right_aspect += 1
                            break;

                elif int(pred_imp_aspect[i]) == 0:
                    for element in value[3]:
                        if element[0] == [0, 0, 0]:
                            flag1 = 1
                            break;
                    if flag1 == 0:
                        pred_imp_right_aspect += 1
                else:
                    print("出错！")
                    return

                if int(pred_imp_opinion[i]) == 1:
                    for element in value[3]:
                        if element[1] == [0, 0, 0]:
                            pred_imp_right_opinion += 1
                            break;

                elif int(pred_imp_opinion[i]) == 0:
                    for element in value[3]:
                        if element[1] == [0, 0, 0]:
                            flag2 = 1
                            break;
                    if flag2 == 0:
                        pred_imp_right_opinion += 1
                else:
                    print("出错！")
                    return
                flag3 = 0
                flag4 = 0
                flag5 = 0
                flag6 = 0
                flag7 = 0
                flag8 = 0
                for element in value[3]:
                    if element[0] == [0, 0, 0] and element[1] != [0, 0, 0]:
                        flag3 = 1
                        if int(pred_imp_aspect[i]) == 1:
                            if flag4 == 0:
                                pred_imp_aspect_num += 1
                                flag4 = 1
                    if element[1] == [0, 0, 0] and element[0] != [0, 0, 0]:
                        flag5 = 1
                        if int(pred_imp_opinion[i]) == 1:
                            if flag6 == 0:
                                pred_imp_opinion_num += 1
                                flag6 = 1
                        else:
                            sentence = " ".join(value[0])
                            if sentence not in opinion_imp_list:
                                opinion_imp_list.append(sentence)
                    if element[1] == [0, 0, 0] and element[0] == [0, 0, 0]:
                        flag7 = 1
                        if int(pred_imp_opinion[i]) == 1 and int(pred_imp_aspect[i]) == 1:
                            if flag8 == 0:
                                pred_imp_asp_opinion_num += 1
                                flag8 = 1
                if flag3 == 1:
                    pred_imp_aspect_num_total += 1
                if flag5 == 1:
                    pred_imp_opinion_num_total += 1
                if flag7 == 1:
                    pred_imp_asp_opinion_num_total += 1
                if flag3 != 1 and flag5 != 1 and flag7!=1:
                    pred_emp_num_total += 1
                pred_imp_aspect_total += 1
                pred_imp_opinion_total += 1



            reverse_pred_stage1_logits = torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2)
            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)
            reverse_pred_stage1_logits = torch.where(spans_mask_tensor == 1, reverse_pred_stage1_logits,
                                             torch.tensor(0).type_as(reverse_pred_stage1_logits))

            aspect_tp,aspect_fp,aspect_fn = cal_step1(spans_ner_label_tensor,pred_aspect_logits,spans_mask_tensor,aspect_tp,aspect_fp,aspect_fn)
            opinion_tp,opinion_fp,opinion_fn = cal_step1(reverse_ner_label_tensor,reverse_pred_stage1_logits,spans_mask_tensor,opinion_tp,opinion_fp,opinion_fn)
            for i in range(len(pred_imp_aspect)):
                pred_aspect_logits[i][0] = pred_imp_aspect[i]
                pred_aspect_logits[i][torch.sum(spans_mask_tensor[i]).item()-1] = 0
                reverse_pred_stage1_logits[i][torch.sum(spans_mask_tensor[i]).item()-1] = pred_imp_opinion[i]
                reverse_pred_stage1_logits[i][0] = 0
            '''真实结果合成'''
            gold_instances.append(dataset.instances[j])
            result = []
            bool_mask = spans_mask_tensor.bool()
            masked_logits = opinion_class_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
            masked_logits_asp = aspect_class_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
            opinion_soft = F.softmax(masked_logits, dim=-2)[:, :, 1]
            aspect_soft = F.softmax(masked_logits_asp, dim=-2)[:, :, 1]
            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] !=0\
                    and torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] != 0:
                opinion_span = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                           torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                aspect_span = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                          torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                # pred_pair = torch.tensor(
                #     [[aspect_span[i][0][1], opinion_span[j][0][1]]
                #      for i in range(len(aspect_span)) for j in range(len(opinion_span))])
                pred_pair = cartesian_product(aspect_span,opinion_span)
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
                        opinion_index =int(span[1])
                        # aspect_rep = forward_embedding[batch,aspect_index]
                        # opinion_rep = reverse_embedding[batch, opinion_index]
                        aspect_rep = spans_embedding[batch, aspect_index]
                        opinion_rep = spans_embedding[batch, opinion_index]
                        last_index = torch.sum(spans_mask_tensor[batch]).item()-1

                        if aspect_index == 0 and opinion_index != last_index:
                            left,_ = torch.max(spans_embedding[batch, 1:opinion_index, :], dim=0) if opinion_index != 1 else \
                                (spans_embedding[batch][0],0)
                            right,_ = torch.max(spans_embedding[batch, opinion_index:last_index, :],
                                               dim=0) if opinion_index != last_index else (spans_embedding[batch][last_index],0)
                        elif aspect_index != 0 and opinion_index == last_index:
                            left,_ = torch.max(spans_embedding[batch, 1:aspect_index, :], dim=0) if aspect_index != 1 else \
                            (spans_embedding[batch][0],0)
                            right,_ = torch.max(spans_embedding[batch, aspect_index:last_index, :],
                                               dim=0) if aspect_index != last_index else (spans_embedding[batch][last_index],0)
                        elif aspect_index == 0 and opinion_index == last_index:
                            left = torch.mean(spans_embedding[batch, 1:last_index, :], dim=0)
                            right,_ = torch.max(spans_embedding[batch, 1:last_index, :], dim=0)
                        else:
                            left = torch.mean(spans_embedding[batch, 1:aspect_index, :], dim=0) if aspect_index != 1 else \
                                spans_embedding[batch][0]
                            right = torch.mean(spans_embedding[batch, aspect_index:last_index, :],
                                               dim=0) if aspect_index != last_index else spans_embedding[batch][last_index]
                        input_rep.append(torch.cat((aspect_rep+opinion_rep,left,right),dim=0))

                        # if aspect_index == 0 and opinion_index != last_index:
                        #     left,_ = torch.max(spans_embedding[batch, 1:opinion_index, :],
                        #                         dim=0) if opinion_index != 1 else \
                        #         (spans_embedding[batch][0],0)
                        #     right,_ = torch.max(spans_embedding[batch, opinion_index:last_index, :],
                        #                          dim=0) if opinion_index != last_index else \
                        #         (spans_embedding[batch][last_index],0)
                        # elif aspect_index != 0 and opinion_index == last_index:
                        #     left,_ = torch.max(spans_embedding[batch, 1:aspect_index, :],
                        #                         dim=0) if aspect_index != 1 else \
                        #         (spans_embedding[batch][0],0)
                        #     right,_ = torch.max(spans_embedding[batch, aspect_index:last_index, :],
                        #                          dim=0) if aspect_index != last_index else \
                        #         (spans_embedding[batch][last_index],0)
                        # elif aspect_index == 0 and opinion_index == last_index:
                        #     left = torch.mean(spans_embedding[batch, 1:last_index, :], dim=0)
                        #     right,_ = torch.max(spans_embedding[batch, 1:last_index, :], dim=0)
                        # else:
                        #     left,_ = torch.max(spans_embedding[batch, 1:aspect_index, :],
                        #                         dim=0) if aspect_index != 1 else \
                        #         (spans_embedding[batch][0],0)
                        #     right,_ = torch.max(spans_embedding[batch, aspect_index:last_index, :],
                        #                          dim=0) if aspect_index != last_index else \
                        #         (spans_embedding[batch][last_index],0)
                        # input_rep.append(torch.cat((aspect_rep + opinion_rep, left, right), dim=0))
                        # # input_rep.append(aspect_rep+opinion_rep)
                    input_rep = torch.stack(input_rep)
                    # # 使用广播计算笛卡尔积并拼接
                    # expanded_aspect = aspect_rep.unsqueeze(1)
                    # expanded_opinion = opinion_rep.unsqueeze(0)
                    # expanded_aspect = expanded_aspect.repeat(1, opinion_rep.size(0), 1)
                    # expanded_opinion = expanded_opinion.repeat(aspect_rep.size(0), 1, 1)
                    # cartesian_product1 = torch.cat((expanded_aspect, expanded_opinion), dim=2)
                    # # 转换形状
                    # input_rep  = cartesian_product1.reshape(aspect_rep.size(0) * opinion_rep.size(0), 768 * 2)
                    category_logits,_ = step_3_category(spans_embedding,bert_spans_tensor,input_rep,spans_mask_tensor)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    result = []
                    for i,pair in enumerate(pred_pair):
                        a_o_c_result = [int(pair[0]),int(pair[1]),int(pred_category_logits[i]),int(pair[2])]
                        result.append(a_o_c_result)
                    category_result.append(result)
            else:
                category_result.append([])

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:

                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                forward_stage2_pred_opinion_sentiment_logit.append(
                    torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
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
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1])).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt),dim=0)

                is_aspect = True
                _, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, step2_forward_embedding,all_span_mask,\
                    all_left_tensor,all_right_tensor,opinion_prob_tensor= stage_2_features_generation1(
                    bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    forward_embedding,reverse_embedding,pred_span_aspect_tensor,is_aspect,opinion_soft)

                opinion_class_logits = step_2_forward(all_spans_embedding, all_span_mask,
                                                                all_span_aspect_tensor,opinion_prob_tensor,step2_forward_embedding)
                #step1. 找到所有的opinion_index及其表征
                #step2. 遍历aspect个数，逐个拼接opinion表征，最后输入到step3_category中
                #step3. 添加到result



                aspect_imp_result = []

                '''
                for index in imp_index:
                    batch_num = pred_aspect_spans[int(index)].squeeze()[0]
                    aspect_span_num = bert_spans_tensor[batch_num, pred_aspect_spans[index].squeeze()[1], :2]
                    opinion_span_num = bert_spans_tensor[batch_num, torch.sum(all_span_mask[index]).item()-1, :2]
                    cat_input_rep = all_span_aspect_tensor[index].squeeze() + all_spans_embedding[index].squeeze(0)[torch.sum(all_span_mask[index]).item()-1]
                    cat_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, cat_input_rep)
                    cat_result = torch.argmax(F.softmax(cat_logits, dim=0), dim=0)
                    aspect_imp_result.append([int(batch_num),aspect_span_num,opinion_span_num,int(cat_result),
                                              int(aspect_imp_polarity[index])])
                '''
                aspect_imp_opinion_result.append(aspect_imp_result)
                '''
                output_values, output_indices = torch.max(opinion_class_logits, dim=-1)
                opinion_list = []
                batch_list = []
                for k,row in enumerate(output_indices):
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    opinion_list.append(non_three_indices_row)
                    batch_list.append(int(pred_aspect_spans[k][0][0]))
                for i, row_indices in enumerate(opinion_list):
                    # print(f"hhh第{i + 1}行不等于3的索引位置：")
                    if row_indices.size(0) == 0:
                        continue
                    opinion_rep = []
                    k = 0
                    for l,index in enumerate(row_indices):
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        if index >= torch.sum(all_span_mask[i]).item():
                            break

                        # opinion_rep.append(spans_embedding[int(pred_aspect_spans[i][0][0])][index.item()])
                        opinion_rep.append(all_spans_embedding[i][index.item()])

                        if math.isinf(spans_embedding[int(pred_aspect_spans[i][0][0])][index.item()][0]):
                            print("inf error")
                        k += 1
                    if has_empty_tensor(opinion_rep):
                        print("empty error")
                    if opinion_rep != []:
                        opinion_rep = torch.stack(opinion_rep)
                    aspect_rep = all_span_aspect_tensor[i][0].clone()
                    aspect_repx3 = aspect_rep.expand(k,-1).to(args.device)
                    # final_rep = torch.cat((aspect_repx3,opinion_rep),dim=1)
                    final_rep = aspect_repx3 + opinion_rep
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j,index in enumerate(row_indices):
                        if index >= torch.sum(all_span_mask[i]).item():
                            break
                        result.append([int(pred_aspect_spans[i][0][1]),int(index),int(pred_category_logits[j]),int(pred_aspect_spans[i][0][0])])
                    '''


                forward_stage1_pred_aspect_result.append(pred_span_aspect_tensor)
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2))
                forward_stage2_pred_opinion_sentiment_logit.append(F.softmax(opinion_class_logits, dim=2))
                # forward_pred_category_logit.append(pred_category_logits)
            '''反向预测'''
            if torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] == 0:
                reverse_stage1_pred_opinion_result.append(torch.full_like(reverse_opinion_tensor, -1))
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.full_like(reverse_aspect_label_tensor, -1))
                reverse_stage2_pred_aspect_sentiment_logit.append(
                    torch.full_like(reverse_aspect_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
                opinion_imp_aspect_result.append([])
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                    # if int(reverse_pred_opinion_span.squeeze()[1]) == 0:
                    #    continue
                    reverse_opinion_tensor_unspilt = bert_spans_tensor[batch_num, reverse_pred_opinion_span.squeeze()[1], :2]
                    reverse_opinion_tensor_unspilt = torch.tensor(
                        (batch_num, reverse_opinion_tensor_unspilt[0], reverse_opinion_tensor_unspilt[1])).unsqueeze(0)
                    if reverse_span_opinion_tensor is None:
                        reverse_span_opinion_tensor = reverse_opinion_tensor_unspilt
                    else:
                        reverse_span_opinion_tensor = torch.cat((reverse_span_opinion_tensor, reverse_opinion_tensor_unspilt), dim=0)
                # __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                # reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_features.last_hidden_state,
                #                                                                          attention_mask,
                #                                                                          bert_spans_tensor,
                #                                                                          spans_mask_tensor,
                #                                                                          spans_embedding,
                #                                                                          reverse_span_opinion_tensor)
                is_aspect=False
                __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, step2_reverse_embedding,reverse_span_mask,all_left_tensor,all_right_tensor,aspect_prob_tensor = stage_2_features_generation1(
                    bert_features.last_hidden_state,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    reverse_embedding,
                    forward_embedding,
                    reverse_span_opinion_tensor,
                    is_aspect,
                    aspect_soft
                    )

                reverse_aspect_class_logits = step_2_reverse(reverse_spans_embedding,reverse_span_mask,all_reverse_opinion_tensor,aspect_prob_tensor,step2_reverse_embedding)




                opinion_imp_result = []
                '''
                for index in imp_index:
                    batch_num = reverse_pred_opinion_spans[index].squeeze()[0]
                    opinion_span_num = bert_spans_tensor[batch_num, reverse_pred_opinion_spans[index].squeeze()[1], :2]
                    aspect_span_num = bert_spans_tensor[batch_num,0, :2]
                    cat_input_rep = all_reverse_opinion_tensor[index].squeeze() + reverse_spans_embedding[index].squeeze()[torch.sum(reverse_span_mask[index]).item()-1]
                    cat_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, cat_input_rep)
                    cat_result = torch.argmax(F.softmax(cat_logits, dim=0), dim=0)
                    opinion_imp_result.append([int(batch_num),aspect_span_num,opinion_span_num,int(cat_result),
                                               int(opinion_imp_polarity[index])])
                '''
                opinion_imp_aspect_result.append(opinion_imp_result)

                '''
                output_values, output_indices = torch.max(reverse_aspect_class_logits, dim=-1)
                aspect_list = []
                for row in output_indices:
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    aspect_list.append(non_three_indices_row)

                for i, row_indices in enumerate(aspect_list):
                    # print(f"第{i + 1}行不等于3的索引位置：")
                    if row_indices.size(0) == 0:
                        continue
                    aspect_rep = []
                    k=0
                    for l,index in enumerate(row_indices):
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        if index >= torch.sum(reverse_span_mask[i]).item():
                            break
                        # aspect_rep.append(spans_embedding[int(reverse_pred_opinion_spans[i][0][0])][index.item()])
                        aspect_rep.append(reverse_spans_embedding[i][index.item()])

                        if math.isinf(spans_embedding[int(reverse_pred_opinion_spans[i][0][0])][index.item()][0]):
                            print("inf error")
                        k += 1
                    if has_empty_tensor(aspect_rep):
                        print("empty error")
                    if aspect_rep != []:
                        aspect_rep = torch.stack(aspect_rep)
                    opinion_rep = all_reverse_opinion_tensor[i][0].clone()
                    opinion_repx3 = opinion_rep.expand(k, -1).to(args.device)
                    # final_rep = torch.cat((aspect_rep, opinion_repx3), dim=1)
                    final_rep = aspect_rep + opinion_repx3
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j, index in enumerate(row_indices):
                        if index >= torch.sum(reverse_span_mask[i]).item():
                            break
                        result.append([int(index), int(reverse_pred_opinion_spans[i][0][1]), int(pred_category_logits[j]),int(reverse_pred_opinion_spans[i][0][0])])
                    '''
                reverse_stage1_pred_opinion_result.append(reverse_span_opinion_tensor)
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.argmax(F.softmax(reverse_aspect_class_logits, dim=2), dim=2))
                reverse_stage2_pred_aspect_sentiment_logit.append(F.softmax(reverse_aspect_class_logits, dim=2))
            # category_result.append(result)
        gold_instances = [x for i in gold_instances for x in i]
        forward_pred_data = (forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment,
                             forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result,
                             forward_stage2_pred_opinion_sentiment_logit,aspect_imp_opinion_result,category_result)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit,opinion_imp_aspect_result)
        reverse_pred_result = unbatch_data(reverse_pred_data)

        metric = Metric(args, forward_pred_result, reverse_pred_result, gold_instances)
        quad_result= metric.score_triples()
        # print('aspect precision:', aspect_result[0], "aspect recall: ", aspect_result[1], "aspect f1: ", aspect_result[2])
        # print('opinion precision:', opinion_result[0], "opinion recall: ", opinion_result[1], "opinion f1: ",
        #       opinion_result[2])
        # print('APCE precision:', apce_result[0], "APCE recall: ", apce_result[1], "APCE f1: ",
        #       apce_result[2])
        # print('pair precision:', pair_result[0], "pair recall:", pair_result[1], "pair f1:", pair_result[2])
        # print('triple precision:', triplet_result[0], "triple recall: ", triplet_result[1], "triple f1: ", triplet_result[2])
        # logger.info(
        #     'aspect precision: {}\taspect recall: {:.8f}\taspect f1: {:.8f}'.format(aspect_result[0], aspect_result[1], aspect_result[2]))
        # logger.info(
        #     'opinion precision: {}\topinion recall: {:.8f}\topinion f1: {:.8f}'.format(opinion_result[0],
        #                                                                                 opinion_result[1],
        #                                                                                 opinion_result[2]))
        # logger.info('APCE precision: {}\tAPCE recall: {:.8f}\tAPCE f1: {:.8f}'.format(apce_result[0],
        #                                                                         apce_result[1], apce_result[2]))
        # logger.info('pair precision: {}\tpair recall: {:.8f}\tpair f1: {:.8f}'.format(pair_result[0],
        #                                                                                   pair_result[1],
        #                                                                                   pair_result[2]))
        # logger.info('triple precision: {}\ttriple recall: {:.8f}\ttriple f1: {:.8f}'.format(triplet_result[0],
        #                                                                                   triplet_result[1],
        #                                                                                   triplet_result[2]))
        aspect_p,aspect_r,aspect_f = cal_f1(aspect_tp,aspect_fp,aspect_fn)
        opinion_p,opinion_r,opinion_f = cal_f1(opinion_tp,opinion_fp,opinion_fn)
        logger.info('step1预测aspect词性能，precision：{}，recall：{}，F1：{}'.format(aspect_p,aspect_r,aspect_f))
        logger.info('step1预测opinion词性能，precision：{}，recall：{}，F1：{}'.format(opinion_p, opinion_r, opinion_f))
        logger.info('预测显式或隐式aspect({}/{})正确率：{:.8f},预测显式或隐式opinion({}/{})正确率：{:.8f}'.format(pred_imp_right_aspect,pred_imp_aspect_total,
                                                                                                pred_imp_right_aspect/pred_imp_aspect_total,
                                                                                    pred_imp_right_opinion, pred_imp_opinion_total,
                                                                                    pred_imp_right_opinion/pred_imp_opinion_total))
        logger.info('单隐式aspect({}/{})正确率：{:.8f},单隐式opinion({}/{})正确率：{:.8f},，双隐式({}/{})正确率{:.8f}:显式共：{}'.format(pred_imp_aspect_num,pred_imp_aspect_num_total,
            pred_imp_aspect_num/ pred_imp_aspect_num_total,pred_imp_opinion_num,pred_imp_opinion_num_total,
            pred_imp_opinion_num / pred_imp_opinion_num_total,pred_imp_asp_opinion_num,pred_imp_asp_opinion_num_total,
                        pred_imp_asp_opinion_num/pred_imp_asp_opinion_num_total,pred_emp_num_total))
        if pred_imp_opinion_num / pred_imp_opinion_num_total < 0.60:
            opinion_imp_list = []
        logger.info('quad precision: {}\tquad recall: {:.8f}\tquad f1: {:.8f}'.format(quad_result[0],
                                                                                            quad_result[1],
                                                                                            quad_result[2]))

    bert_model.train()
    step_1_model.train()
    step_2_forward.train()
    step_2_reverse.train()
    step_3_category.train()
    return quad_result,opinion_imp_list

def has_empty_tensor(tensor_list):
    for tensor in tensor_list:
        if tensor.numel() == 0:
            return True
    return False


def cal_softmax(step_2_logits, all_span_mask):
    bool_mask = all_span_mask.bool()

    masked_logits = step_2_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
    soft = F.softmax(masked_logits, dim=1)[:, :, 1:]
    final_soft = torch.sum(soft,dim=-1)/len(soft)
    return final_soft


def eval1(bert_model, step_1_model, step_2_forward, step_2_reverse,step_3_category, dataset, args):
    with torch.no_grad():
        bert_model.eval()
        step_1_model.eval()
        step_2_forward.eval()
        step_2_reverse.eval()
        step_3_category.eval()
        '''真实结果'''
        gold_instances = []
        '''前向预测结果'''
        forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment, \
        forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result, \
        forward_stage2_pred_opinion_sentiment_logit,forward_pred_category_logit = [],[],[],[],[],[]
        category_result = []
        '''反向预测结果'''
        reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment, \
        reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result, \
        reverse_stage2_pred_aspect_sentiment_logit = [], [], [], [], []

        '''隐式通道预测结果'''
        aspect_imp_opinion_result,opinion_imp_aspect_result = [],[]
        is_aspect = False
        pred_imp_right_aspect = 0
        pred_imp_right_opinion = 0
        pred_imp_aspect_total = 0
        pred_imp_opinion_total = 0

        pred_imp_aspect_num = 0
        pred_imp_opinion_num = 0
        pred_imp_asp_opinion_num = 0
        pred_imp_aspect_num_total = 0
        pred_imp_opinion_num_total = 0
        pred_imp_asp_opinion_num_total = 0
        pred_emp_num_total = 0
        opinion_imp_list = []
        aspect_tp = 0
        aspect_fp = 0
        aspect_fn = 0

        opinion_tp = 0
        opinion_fp = 0
        opinion_fn = 0
        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
            reverse_aspect_label_tensor, related_spans_tensor, sentence_length, \
            imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, \
            opinion_polarity_label_tensor = dataset.get_batch(j)


            bert_features = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)

            # if j == 0 and args.model_para_test:
            #     step_1_model.to("cpu")
            #
            #     flop_step1, para_step1 = profile(step_1_model, inputs=(bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
            #                                                          related_spans_tensor, sentence_length))
            #     macs, param = clever_format([flop_step1,para_step1], "%.3f")
            #     print("STEP 1 MACs: ", macs, "STEP 1 Params", param)
            #     logger.info(
            #         'STEP 1 MACs:  {}\tSTEP 1 Params: {:.8f}\n\n'.format(macs, param))

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
            cnn_spans_mask_tensor,imp_aspect_exist,imp_opinion_exist = step_1_model(
                bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                related_spans_tensor, bert_features.pooler_output,sentence_length)



            '''Batch更新'''
            pred_aspect_logits_1 = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits_1 = torch.where(spans_mask_tensor == 1, pred_aspect_logits_1,
                                             torch.tensor(0).type_as(pred_aspect_logits_1))
            pred_aspect_logits = torch.zeros_like(pred_aspect_logits_1)
            pred_aspect_logits[pred_aspect_logits_1 == 1] = 1
            reverse_pred_stage1_logits = torch.zeros_like(pred_aspect_logits_1)
            reverse_pred_stage1_logits[pred_aspect_logits_1 == 2] = 1

            pred_imp_aspect = torch.argmax(F.softmax(imp_aspect_exist,dim=1),dim=1)
            pred_imp_opinion = torch.argmax(F.softmax(imp_opinion_exist,dim=1),dim=1)

            # pred_category_logits = torch.where(spans_mask_tensor == 1,pred_category_logits,
            #                                    torch.tensor(0).type_as(pred_category_logits))

            for i, value in enumerate(sentence_length):
                flag1 = 0
                flag2 = 0
                if int(pred_imp_aspect[i]) == 1:
                    for element in value[3]:
                        if element[0] == [0, 0, 0]:
                            pred_imp_right_aspect += 1
                            break

                elif int(pred_imp_aspect[i]) == 0:
                    for element in value[3]:
                        if element[0] == [0, 0, 0]:
                            flag1 = 1
                            break
                    if flag1 == 0:
                        pred_imp_right_aspect += 1
                else:
                    print("出错！")
                    return

                if int(pred_imp_opinion[i]) == 1:
                    for element in value[3]:
                        if element[1] == [0, 0, 0]:
                            pred_imp_right_opinion += 1
                            break

                elif int(pred_imp_opinion[i]) == 0:
                    for element in value[3]:
                        if element[1] == [0, 0, 0]:
                            flag2 = 1
                            break
                    if flag2 == 0:
                        pred_imp_right_opinion += 1
                else:
                    print("出错！")
                    return
                flag3 = 0
                flag4 = 0
                flag5 = 0
                flag6 = 0
                flag7 = 0
                flag8 = 0
                for element in value[3]:
                    if element[0] == [0, 0, 0] and element[1] != [0, 0, 0]:
                        flag3 = 1
                        if int(pred_imp_aspect[i]) == 1:
                            if flag4 == 0:
                                pred_imp_aspect_num += 1
                                flag4 = 1
                    if element[1] == [0, 0, 0] and element[0] != [0, 0, 0]:
                        flag5 = 1
                        if int(pred_imp_opinion[i]) == 1:
                            if flag6 == 0:
                                pred_imp_opinion_num += 1
                                flag6 = 1
                        else:
                            sentence = " ".join(value[0])
                            if sentence not in opinion_imp_list:
                                opinion_imp_list.append(sentence)
                    if element[1] == [0, 0, 0] and element[0] == [0, 0, 0]:
                        flag7 = 1
                        if int(pred_imp_opinion[i]) == 1 and int(pred_imp_aspect[i]) == 1:
                            if flag8 == 0:
                                pred_imp_asp_opinion_num += 1
                                flag8 = 1
                if flag3 == 1:
                    pred_imp_aspect_num_total += 1
                if flag5 == 1:
                    pred_imp_opinion_num_total += 1
                if flag7 == 1:
                    pred_imp_asp_opinion_num_total += 1
                if flag3 != 1 and flag5 != 1 and flag7!=1:
                    pred_emp_num_total += 1
                pred_imp_aspect_total += 1
                pred_imp_opinion_total += 1




            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)


            aspect_tp,aspect_fp,aspect_fn = cal_step1(spans_ner_label_tensor,pred_aspect_logits,spans_mask_tensor,aspect_tp,aspect_fp,aspect_fn)
            opinion_tp,opinion_fp,opinion_fn = cal_step1(reverse_ner_label_tensor,reverse_pred_stage1_logits,spans_mask_tensor,opinion_tp,opinion_fp,opinion_fn)
            for i in range(len(pred_imp_aspect)):
                pred_aspect_logits[i][0] = pred_imp_aspect[i]
                pred_aspect_logits[i][torch.sum(spans_mask_tensor[i]).item() - 1] = 0
                reverse_pred_stage1_logits[i][torch.sum(spans_mask_tensor[i]).item() - 1] = pred_imp_opinion[i]
                reverse_pred_stage1_logits[i][0] = 0
            '''真实结果合成'''
            gold_instances.append(dataset.instances[j])
            result = []
            bool_mask = spans_mask_tensor.bool()
            masked_logits = opinion_class_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
            masked_logits_asp = aspect_class_logits.masked_fill(~bool_mask.unsqueeze(-1), float('-inf'))
            opinion_soft = F.softmax(masked_logits, dim=-2)[:, :, 1]
            aspect_soft = F.softmax(masked_logits_asp, dim=-2)[:, :, 1]
            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] !=0\
                    and torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] != 0:
                opinion_span = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                           torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                aspect_span = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                          torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                # pred_pair = torch.tensor(
                #     [[aspect_span[i][0][1], opinion_span[j][0][1]]
                #      for i in range(len(aspect_span)) for j in range(len(opinion_span))])
                pred_pair = cartesian_product(aspect_span,opinion_span)
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
                        opinion_index =int(span[1])
                        # aspect_rep = forward_embedding[batch,aspect_index]
                        # opinion_rep = reverse_embedding[batch, opinion_index]
                        aspect_rep = spans_embedding[batch, aspect_index]
                        opinion_rep = spans_embedding[batch, opinion_index]
                        last_index = torch.sum(spans_mask_tensor[batch]).item()-1
                        if aspect_index == 0 and opinion_index != last_index:
                            left,_ = torch.max(spans_embedding[batch, 1:opinion_index, :], dim=0) if opinion_index != 1 else \
                                (spans_embedding[batch][0],0)
                            right,_ = torch.max(spans_embedding[batch, opinion_index:last_index, :],
                                               dim=0) if opinion_index != last_index else (spans_embedding[batch][last_index],0)
                        elif aspect_index != 0 and opinion_index == last_index:
                            left,_ = torch.max(spans_embedding[batch, 1:aspect_index, :], dim=0) if aspect_index != 1 else \
                            (spans_embedding[batch][0],0)
                            right,_ = torch.max(spans_embedding[batch, aspect_index:last_index, :],
                                               dim=0) if aspect_index != last_index else (spans_embedding[batch][last_index],0)
                        elif aspect_index == 0 and opinion_index == last_index:
                            left = torch.mean(spans_embedding[batch, 1:last_index, :], dim=0)
                            right,_ = torch.max(spans_embedding[batch, 1:last_index, :], dim=0)
                        else:
                            left,_ = torch.max(spans_embedding[batch, 1:aspect_index, :], dim=0) if aspect_index != 1 else \
                           (spans_embedding[batch][0],0)
                            right,_ = torch.max(spans_embedding[batch, aspect_index:last_index, :],
                                               dim=0) if aspect_index != last_index else (spans_embedding[batch][last_index],0)
                        input_rep.append(torch.cat((aspect_rep+opinion_rep,left,right),dim=0))
                        # input_rep.append(aspect_rep+opinion_rep)
                    input_rep = torch.stack(input_rep)
                    # # 使用广播计算笛卡尔积并拼接
                    # expanded_aspect = aspect_rep.unsqueeze(1)
                    # expanded_opinion = opinion_rep.unsqueeze(0)
                    # expanded_aspect = expanded_aspect.repeat(1, opinion_rep.size(0), 1)
                    # expanded_opinion = expanded_opinion.repeat(aspect_rep.size(0), 1, 1)
                    # cartesian_product1 = torch.cat((expanded_aspect, expanded_opinion), dim=2)
                    # # 转换形状
                    # input_rep  = cartesian_product1.reshape(aspect_rep.size(0) * opinion_rep.size(0), 768 * 2)
                    category_logits,_ = step_3_category(spans_embedding,bert_spans_tensor,input_rep,spans_mask_tensor)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    result = []
                    for i,pair in enumerate(pred_pair):
                        a_o_c_result = [int(pair[0]),int(pair[1]),int(pred_category_logits[i]),int(pair[2])]
                        result.append(a_o_c_result)
                    category_result.append(result)
            else:
                category_result.append([])

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:

                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                forward_stage2_pred_opinion_sentiment_logit.append(
                    torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
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
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1])).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt),dim=0)

                is_aspect = True
                _, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, step2_forward_embedding,all_span_mask,\
                    all_left_tensor,all_right_tensor,opinion_prob_tensor= stage_2_features_generation1(
                    bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    forward_embedding,reverse_embedding,pred_span_aspect_tensor,is_aspect,opinion_soft)

                opinion_class_logits = step_2_forward(all_spans_embedding, all_span_mask,
                                                                all_span_aspect_tensor,opinion_prob_tensor,step2_forward_embedding)
                #step1. 找到所有的opinion_index及其表征
                #step2. 遍历aspect个数，逐个拼接opinion表征，最后输入到step3_category中
                #step3. 添加到result



                aspect_imp_result = []

                '''
                for index in imp_index:
                    batch_num = pred_aspect_spans[int(index)].squeeze()[0]
                    aspect_span_num = bert_spans_tensor[batch_num, pred_aspect_spans[index].squeeze()[1], :2]
                    opinion_span_num = bert_spans_tensor[batch_num, torch.sum(all_span_mask[index]).item()-1, :2]
                    cat_input_rep = all_span_aspect_tensor[index].squeeze() + all_spans_embedding[index].squeeze(0)[torch.sum(all_span_mask[index]).item()-1]
                    cat_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, cat_input_rep)
                    cat_result = torch.argmax(F.softmax(cat_logits, dim=0), dim=0)
                    aspect_imp_result.append([int(batch_num),aspect_span_num,opinion_span_num,int(cat_result),
                                              int(aspect_imp_polarity[index])])
                '''
                aspect_imp_opinion_result.append(aspect_imp_result)
                '''
                output_values, output_indices = torch.max(opinion_class_logits, dim=-1)
                opinion_list = []
                batch_list = []
                for k,row in enumerate(output_indices):
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    opinion_list.append(non_three_indices_row)
                    batch_list.append(int(pred_aspect_spans[k][0][0]))
                for i, row_indices in enumerate(opinion_list):
                    # print(f"hhh第{i + 1}行不等于3的索引位置：")
                    if row_indices.size(0) == 0:
                        continue
                    opinion_rep = []
                    k = 0
                    for l,index in enumerate(row_indices):
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        if index >= torch.sum(all_span_mask[i]).item():
                            break

                        # opinion_rep.append(spans_embedding[int(pred_aspect_spans[i][0][0])][index.item()])
                        opinion_rep.append(all_spans_embedding[i][index.item()])

                        if math.isinf(spans_embedding[int(pred_aspect_spans[i][0][0])][index.item()][0]):
                            print("inf error")
                        k += 1
                    if has_empty_tensor(opinion_rep):
                        print("empty error")
                    if opinion_rep != []:
                        opinion_rep = torch.stack(opinion_rep)
                    aspect_rep = all_span_aspect_tensor[i][0].clone()
                    aspect_repx3 = aspect_rep.expand(k,-1).to(args.device)
                    # final_rep = torch.cat((aspect_repx3,opinion_rep),dim=1)
                    final_rep = aspect_repx3 + opinion_rep
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j,index in enumerate(row_indices):
                        if index >= torch.sum(all_span_mask[i]).item():
                            break
                        result.append([int(pred_aspect_spans[i][0][1]),int(index),int(pred_category_logits[j]),int(pred_aspect_spans[i][0][0])])
                    '''


                forward_stage1_pred_aspect_result.append(pred_span_aspect_tensor)
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2))
                forward_stage2_pred_opinion_sentiment_logit.append(F.softmax(opinion_class_logits, dim=2))
                # forward_pred_category_logit.append(pred_category_logits)
            '''反向预测'''
            if torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] == 0:
                reverse_stage1_pred_opinion_result.append(torch.full_like(reverse_opinion_tensor, -1))
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.full_like(reverse_aspect_label_tensor, -1))
                reverse_stage2_pred_aspect_sentiment_logit.append(
                    torch.full_like(reverse_aspect_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
                opinion_imp_aspect_result.append([])
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                    # if int(reverse_pred_opinion_span.squeeze()[1]) == 0:
                    #    continue
                    reverse_opinion_tensor_unspilt = bert_spans_tensor[batch_num, reverse_pred_opinion_span.squeeze()[1], :2]
                    reverse_opinion_tensor_unspilt = torch.tensor(
                        (batch_num, reverse_opinion_tensor_unspilt[0], reverse_opinion_tensor_unspilt[1])).unsqueeze(0)
                    if reverse_span_opinion_tensor is None:
                        reverse_span_opinion_tensor = reverse_opinion_tensor_unspilt
                    else:
                        reverse_span_opinion_tensor = torch.cat((reverse_span_opinion_tensor, reverse_opinion_tensor_unspilt), dim=0)
                # __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                # reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_features.last_hidden_state,
                #                                                                          attention_mask,
                #                                                                          bert_spans_tensor,
                #                                                                          spans_mask_tensor,
                #                                                                          spans_embedding,
                #                                                                          reverse_span_opinion_tensor)
                is_aspect=False
                __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, step2_reverse_embedding,reverse_span_mask,all_left_tensor,all_right_tensor,aspect_prob_tensor = stage_2_features_generation1(
                    bert_features.last_hidden_state,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    reverse_embedding,
                    forward_embedding,
                    reverse_span_opinion_tensor,
                    is_aspect,
                    aspect_soft
                    )

                reverse_aspect_class_logits = step_2_reverse(reverse_spans_embedding,reverse_span_mask,all_reverse_opinion_tensor,aspect_prob_tensor,step2_reverse_embedding)




                opinion_imp_result = []
                '''
                for index in imp_index:
                    batch_num = reverse_pred_opinion_spans[index].squeeze()[0]
                    opinion_span_num = bert_spans_tensor[batch_num, reverse_pred_opinion_spans[index].squeeze()[1], :2]
                    aspect_span_num = bert_spans_tensor[batch_num,0, :2]
                    cat_input_rep = all_reverse_opinion_tensor[index].squeeze() + reverse_spans_embedding[index].squeeze()[torch.sum(reverse_span_mask[index]).item()-1]
                    cat_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, cat_input_rep)
                    cat_result = torch.argmax(F.softmax(cat_logits, dim=0), dim=0)
                    opinion_imp_result.append([int(batch_num),aspect_span_num,opinion_span_num,int(cat_result),
                                               int(opinion_imp_polarity[index])])
                '''
                opinion_imp_aspect_result.append(opinion_imp_result)

                '''
                output_values, output_indices = torch.max(reverse_aspect_class_logits, dim=-1)
                aspect_list = []
                for row in output_indices:
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    aspect_list.append(non_three_indices_row)

                for i, row_indices in enumerate(aspect_list):
                    # print(f"第{i + 1}行不等于3的索引位置：")
                    if row_indices.size(0) == 0:
                        continue
                    aspect_rep = []
                    k=0
                    for l,index in enumerate(row_indices):
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        if index >= torch.sum(reverse_span_mask[i]).item():
                            break
                        # aspect_rep.append(spans_embedding[int(reverse_pred_opinion_spans[i][0][0])][index.item()])
                        aspect_rep.append(reverse_spans_embedding[i][index.item()])

                        if math.isinf(spans_embedding[int(reverse_pred_opinion_spans[i][0][0])][index.item()][0]):
                            print("inf error")
                        k += 1
                    if has_empty_tensor(aspect_rep):
                        print("empty error")
                    if aspect_rep != []:
                        aspect_rep = torch.stack(aspect_rep)
                    opinion_rep = all_reverse_opinion_tensor[i][0].clone()
                    opinion_repx3 = opinion_rep.expand(k, -1).to(args.device)
                    # final_rep = torch.cat((aspect_rep, opinion_repx3), dim=1)
                    final_rep = aspect_rep + opinion_repx3
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j, index in enumerate(row_indices):
                        if index >= torch.sum(reverse_span_mask[i]).item():
                            break
                        result.append([int(index), int(reverse_pred_opinion_spans[i][0][1]), int(pred_category_logits[j]),int(reverse_pred_opinion_spans[i][0][0])])
                    '''
                reverse_stage1_pred_opinion_result.append(reverse_span_opinion_tensor)
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.argmax(F.softmax(reverse_aspect_class_logits, dim=2), dim=2))
                reverse_stage2_pred_aspect_sentiment_logit.append(F.softmax(reverse_aspect_class_logits, dim=2))
            # category_result.append(result)
        gold_instances = [x for i in gold_instances for x in i]
        forward_pred_data = (forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment,
                             forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result,
                             forward_stage2_pred_opinion_sentiment_logit,aspect_imp_opinion_result,category_result)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit,opinion_imp_aspect_result)
        reverse_pred_result = unbatch_data(reverse_pred_data)

        metric = Metric(args, forward_pred_result, reverse_pred_result, gold_instances)
        quad_result= metric.score_triples()
        # print('aspect precision:', aspect_result[0], "aspect recall: ", aspect_result[1], "aspect f1: ", aspect_result[2])
        # print('opinion precision:', opinion_result[0], "opinion recall: ", opinion_result[1], "opinion f1: ",
        #       opinion_result[2])
        # print('APCE precision:', apce_result[0], "APCE recall: ", apce_result[1], "APCE f1: ",
        #       apce_result[2])
        # print('pair precision:', pair_result[0], "pair recall:", pair_result[1], "pair f1:", pair_result[2])
        # print('triple precision:', triplet_result[0], "triple recall: ", triplet_result[1], "triple f1: ", triplet_result[2])
        # logger.info(
        #     'aspect precision: {}\taspect recall: {:.8f}\taspect f1: {:.8f}'.format(aspect_result[0], aspect_result[1], aspect_result[2]))
        # logger.info(
        #     'opinion precision: {}\topinion recall: {:.8f}\topinion f1: {:.8f}'.format(opinion_result[0],
        #                                                                                 opinion_result[1],
        #                                                                                 opinion_result[2]))
        # logger.info('APCE precision: {}\tAPCE recall: {:.8f}\tAPCE f1: {:.8f}'.format(apce_result[0],
        #                                                                         apce_result[1], apce_result[2]))
        # logger.info('pair precision: {}\tpair recall: {:.8f}\tpair f1: {:.8f}'.format(pair_result[0],
        #                                                                                   pair_result[1],
        #                                                                                   pair_result[2]))
        # logger.info('triple precision: {}\ttriple recall: {:.8f}\ttriple f1: {:.8f}'.format(triplet_result[0],
        #                                                                                   triplet_result[1],
        #                                                                                   triplet_result[2]))
        aspect_p,aspect_r,aspect_f = cal_f1(aspect_tp,aspect_fp,aspect_fn)
        opinion_p,opinion_r,opinion_f = cal_f1(opinion_tp,opinion_fp,opinion_fn)
        logger.info('step1预测aspect词性能，precision：{}，recall：{}，F1：{}'.format(aspect_p,aspect_r,aspect_f))
        logger.info('step1预测opinion词性能，precision：{}，recall：{}，F1：{}'.format(opinion_p, opinion_r, opinion_f))
        logger.info('预测显式或隐式aspect({}/{})正确率：{:.8f},预测显式或隐式opinion({}/{})正确率：{:.8f}'.format(pred_imp_right_aspect,pred_imp_aspect_total,
                                                                                                pred_imp_right_aspect/pred_imp_aspect_total,
                                                                                    pred_imp_right_opinion, pred_imp_opinion_total,
                                                                                    pred_imp_right_opinion/pred_imp_opinion_total))
        logger.info('单隐式aspect({}/{})正确率：{:.8f},单隐式opinion({}/{})正确率：{:.8f},，双隐式({}/{})正确率{:.8f}:显式共：{}'.format(pred_imp_aspect_num,pred_imp_aspect_num_total,
            pred_imp_aspect_num/ pred_imp_aspect_num_total,pred_imp_opinion_num,pred_imp_opinion_num_total,
            pred_imp_opinion_num / pred_imp_opinion_num_total,pred_imp_asp_opinion_num,pred_imp_asp_opinion_num_total,
                        pred_imp_asp_opinion_num/pred_imp_asp_opinion_num_total,pred_emp_num_total))
        if pred_imp_opinion_num / pred_imp_opinion_num_total < 0.60:
            opinion_imp_list = []
        logger.info('quad precision: {}\tquad recall: {:.8f}\tquad f1: {:.8f}'.format(quad_result[0],
                                                                                            quad_result[1],
                                                                                            quad_result[2]))

    bert_model.train()
    step_1_model.train()
    step_2_forward.train()
    step_2_reverse.train()
    step_3_category.train()
    return quad_result,opinion_imp_list
def train(args):
    #print(torch.cuda.device_count())
    #print(torch.cuda.is_available())

    torch.cuda.current_device()
    torch.cuda._initialized = True
    if args.dataset == 'restaurant':
        train_path = "./datasets/Restaurant-ACOS/rest16_quad_train.tsv"
        test_path = "./datasets/Restaurant-ACOS/rest16_quad_test.tsv"
        dev_path = "./datasets/Restaurant-ACOS/rest16_quad_dev.tsv"
    else:
        train_path = "./datasets/Laptop-ACOS/laptop_quad_train.tsv"
        test_path = "./datasets/Laptop-ACOS/laptop_quad_test.tsv"
        dev_path = "./datasets/Laptop-ACOS/laptop_quad_dev.tsv"
    print('-------------------------------')
    print('开始加载测试集')
    logger.info('开始加载测试集')
    # 从指定路径加载测试数据集(args,path:if_train)
    test_datasets = load_data1(args, test_path, if_train=False)
    testset = DataTterator2(test_datasets, args)
    print('测试集加载完成')
    logger.info('测试集加载完成')
    print('-------------------------------')

    Bert = BertModel.from_pretrained(args.init_model)

    bert_config = Bert.config
    Bert.to(args.device)
    # 获取bert
    bert_param_optimizer = list(Bert.named_parameters())

    step_1_model = Step_1(args, bert_config)
    step_1_model.to(args.device)
    step_1_param_optimizer = list(step_1_model.named_parameters())

    step_3_category = Step_3_categories(args)
    step_3_category.to(args.device)
    step_3_category_optimizer  = list(step_3_category.named_parameters())

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
    optimizer = AdamW(training_param_optimizer, lr=args.learning_rate)

    if args.muti_gpu:
        Bert = torch.nn.DataParallel(Bert)
        step_1_model = torch.nn.DataParallel(step_1_model)
        step2_forward_model = torch.nn.DataParallel(step2_forward_model)
        step2_reverse_model = torch.nn.DataParallel(step2_reverse_model)
        step_3_category = torch.nn.DataParallel(step_3_category)

    if args.mode == 'train':
        print('-------------------------------')
        logger.info('开始加载训练与验证集')
        print('开始加载训练与验证集')
        train_datasets = load_data1(args, train_path, if_train=False)
        trainset = DataTterator2(train_datasets, args)
        print("Train features build completed")

        print("Dev features build beginning")
        # dev_datasets = load_data1(args, dev_path, if_train=False)
        # devset = DataTterator2(dev_datasets, args)
        print('训练集与验证集加载完成')
        logger.info('训练集与验证集加载完成')
        print('-------------------------------')
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # scheduler
        # if args.whether_warm_up:
        #     training_steps = args.epochs * devset.batch_count
        #     warmup_steps = int(training_steps * args.warm_up)
        #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
        #                                                 num_training_steps=training_steps)

        tot_loss = 0
        tot_kl_loss = 0
        best_aspect_f1, best_opinion_f1, best_APCE_f1, best_pairs_f1, best_quad_f1,best_quad_precision,best_quad_recall = 0,0,0,0,0,0,0
        best_quad_epoch= 0
        opinion_all_list = []
        num = 0
        for i in range(args.epochs):
            logger.info(('Epoch:{}'.format(i)))

            for j in tqdm.trange(trainset.batch_count):
            # for j in range(trainset.batch_count):
                if j == 1:
                    start = time.time()
                # if j == 1:
                #     print()
                optimizer.zero_grad()

                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length,\
                imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, \
                opinion_polarity_label_tensor = trainset.get_batch(j)
                '''
                for batch_nums,opinion_value in enumerate(reverse_ner_label_tensor):
                    for index,value in enumerate(opinion_value):
                        if value.item() ==1:
                            spans_ner_label_tensor[batch_nums][index]=2
                '''
                bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)
                # class_logits_aspect（shape=[batch_size, 2]）
                # class_logits_opinion（shape=[batch_size, 2]）
                # aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                # cnn_spans_mask_tensor= step_1_model(
                #                                                                       bert_output.last_hidden_state,
                #                                                                       attentio n_mask,
                #                                                                       bert_spans_tensor,
                #                                                                       spans_mask_tensor,
                #                                                                       related_spans_tensor,
                #                                                                       sentence_length)
                aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                cnn_spans_mask_tensor,imp_aspect_exist,imp_opinion_exist = step_1_model(
                                                                            bert_output.last_hidden_state,
                                                                            attention_mask,
                                                                            bert_spans_tensor,
                                                                            spans_mask_tensor,
                                                                            related_spans_tensor,
                                                                            bert_output.pooler_output,
                                                                            sentence_length,
                )
                bool_mask = spans_mask_tensor.bool()

                masked_logits = opinion_class_logits.masked_fill(~bool_mask.unsqueeze(-1),float('-inf'))
                masked_logits_asp = aspect_class_logits.masked_fill(~bool_mask.unsqueeze(-1),float('-inf'))
                opinion_soft = F.softmax(masked_logits,dim=-2)[:,:,1]
                aspect_soft = F.softmax(masked_logits_asp,dim=-2)[:,:,1]
                category_labels = [t[4][0] for t in sentence_length]
                pairs = [t[3] for t in sentence_length]
                category_logits,category_label = step_3_category(spans_embedding,bert_spans_tensor,pairs,spans_mask_tensor,category_labels)
                is_aspect = True
                '''Batch更新'''
                # 21X420                     21 1 768           21 100 768           21 100        21 420 768    21 420
                # 从输入的 BERT 特征、注意力掩码、span 序列和 span 掩码中生成经过处理后的特征
                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                all_spans_embedding, step2_forward_embedding,all_span_mask,all_left_forward_tensor,all_right_forward_tensor,opinion_prob_tensor = stage_2_features_generation1(bert_output.last_hidden_state,
                                                                             attention_mask, bert_spans_tensor,
                                                                             spans_mask_tensor, forward_embedding,reverse_embedding,
                                                                             spans_aspect_tensor,
                                                                                is_aspect,opinion_soft,
                                                                             spans_opinion_label_tensor,
                                                                                 )
                is_aspect = False
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, step2_reverse_embedding,reverse_span_mask,all_left_reverse_tensor,all_right_reverse_tensor,aspect_prob_tensor = stage_2_features_generation1(bert_output.last_hidden_state,
                                                                                     attention_mask, bert_spans_tensor,
                                                                                     spans_mask_tensor, reverse_embedding,
                                                                                     forward_embedding,
                                                                                     reverse_opinion_tensor,
                                                                                        is_aspect,
                                                                                     aspect_soft,reverse_aspect_label_tensor
                                                                                     )

                # aspect->opinion
                step_2_opinion_class_logits= \
                    step2_forward_model(all_spans_embedding, all_span_mask, all_span_aspect_tensor,opinion_prob_tensor,step2_forward_embedding)
                # opinion->aspect
                step_2_aspect_class_logits= step2_reverse_model(reverse_spans_embedding,
                    reverse_span_mask, all_reverse_opinion_tensor,aspect_prob_tensor,step2_reverse_embedding)

                forward_softmax = cal_softmax(step_2_opinion_class_logits,all_span_mask)
                reverse_softmax = cal_softmax(step_2_aspect_class_logits,reverse_span_mask)
                exist_aspect = spans_ner_label_tensor[:,0]
                exist_opinion = reverse_ner_label_tensor[range(reverse_ner_label_tensor.shape[0]), torch.sum(spans_mask_tensor, dim=-1) - 1]
                # 计算loss和KL loss
                # loss, kl_loss = Loss(spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor, step_2_opinion_class_logits,
                #             spans_mask_tensor, all_span_mask, reverse_ner_label_tensor, opinion_class_logits,
                #             all_reverse_aspect_tensor, step_2_aspect_class_logits, cnn_spans_mask_tensor, reverse_span_mask,
                #             spans_embedding, related_spans_tensor, args)
                loss, kl_loss = Loss(spans_ner_label_tensor, aspect_class_logits,
                                     all_span_opinion_tensor,step_2_opinion_class_logits,
                                     spans_mask_tensor, all_span_mask,
                                     reverse_ner_label_tensor, opinion_class_logits,
                                     all_reverse_aspect_tensor, step_2_aspect_class_logits,
                                     cnn_spans_mask_tensor,reverse_span_mask,
                                     spans_embedding, related_spans_tensor,
                                     category_label,category_logits,
                                     exist_aspect,exist_opinion,
                                     imp_aspect_exist, imp_opinion_exist,
                                     sentence_length,
                                     opinion_prob_tensor,forward_softmax,
                                     aspect_prob_tensor,reverse_softmax,
                                     args)
                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps
                    loss.backward()
                    if ((j + 1) % args.accumulation_steps) == 0:
                        optimizer.step()
                        if args.whether_warm_up:
                            scheduler.step()
                else:
                    loss.backward()
                    optimizer.step()
                    if args.whether_warm_up:
                        scheduler.step()
                tot_loss += loss.item()
                tot_kl_loss += kl_loss
            logger.info(('Loss:', tot_loss))
            logger.info(('KL_Loss:', tot_kl_loss))
            tot_loss = 0
            tot_kl_loss = 0
            if i == 10:
                print()
            # print('Evaluating, please wait')
            # aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, step_1_model,
            #                                                                                step2_forward_model,
            #                                                                                step2_reverse_model,
            #                                                                                devset, args)
            quad_result,opinion_imp_list = eval(Bert, step_1_model, step2_forward_model, step2_reverse_model,step_3_category, testset, args)
            if opinion_imp_list != []:
                opinion_all_list.append(opinion_imp_list)
            # print('Evaluating complete')
            # if aspect_result[2] > best_aspect_f1:
            #     best_aspect_f1 = aspect_result[2]
            #     best_aspect_precision = aspect_result[0]
            #     best_aspect_recall = aspect_result[1]
            #     best_aspect_epoch = i
            #
            # if opinion_result[2] > best_opinion_f1:
            #     best_opinion_f1 = opinion_result[2]
            #     best_opinion_precision = opinion_result[0]
            #     best_opinion_recall = opinion_result[1]
            #     best_opinion_epoch = i
            #
            # if apce_result[2] > best_APCE_f1:
            #     best_APCE_f1 = apce_result[2]
            #     best_APCE_precision = apce_result[0]
            #     best_APCE_recall = apce_result[1]
            #     best_APCE_epoch = i
            #
            # if pair_result[2] > best_pairs_f1:
            #     best_pairs_f1 = pair_result[2]
            #     best_pairs_precision = pair_result[0]
            #     best_pairs_recall = pair_result[1]
            #     best_pairs_epoch = i
            if len(opinion_all_list) > 10:
                common_sentences = set(opinion_all_list[0])
                # with open("common_strings" + str(num) + ".txt", "w") as file:
                #     for string in common_sentences:
                #         file.write(string + "\n")
                num += 1
                opinion_all_list = []
            if quad_result[2] > best_quad_f1:
                if args.dataset == "laptop":
                    limit = 0.40
                else:
                    limit = 0.56
                if quad_result[2]>limit:
                    model_path = args.model_dir +args.dataset +'_'+ str(quad_result[2]) + '.pt'


                    state = {
                        "bert_model": Bert.state_dict(),
                        "step_1_model": step_1_model.state_dict(),
                        "step2_forward_model": step2_forward_model.state_dict(),
                        "step2_reverse_model": step2_reverse_model.state_dict(),
                        "step_3_category":step_3_category.state_dict(),
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
        # logger.info(
        #     'best aspect epoch: {}\tbest aspect precision: {:.8f}\tbest aspect recall: {:.8f}\tbest aspect f1: {:.8f}'.
        #         format(best_aspect_epoch, best_aspect_precision, best_aspect_recall, best_aspect_f1))
        # logger.info(
        #     'best opinion epoch: {}\tbest opinion precision: {:.8f}\tbest opinion recall: {:.8f}\tbest opinion f1: {:.8f}'.
        #         format(best_opinion_epoch, best_opinion_precision, best_opinion_recall, best_opinion_f1))
        #
        # logger.info('best APCE epoch: {}\tbest APCE precision: {:.8f}\tbest APCE recall: {:.8f}\tbest APCE f1: {:.8f}'.
        #       format(best_APCE_epoch, best_APCE_precision, best_APCE_recall, best_APCE_f1))
        # logger.info('best pair epoch: {}\tbest pair precision: {:.8f}\tbest pair recall: {:.8f}\tbest pair f1: {:.8f}'.
        #       format(best_pairs_epoch, best_pairs_precision, best_pairs_recall, best_pairs_f1))
        logger.info(
            'best quad epoch: {}\tbest quad precision: {:.8f}\tbest quad recall: {:.8f}\tbest quad f1: {:.8f}'.
            format(best_quad_epoch, best_quad_precision, best_quad_recall, best_quad_f1))

    logger.info("Features build completed")
    logger.info("Evaluation on testset:")

    # model_path = args.model_dir + args.dataset + '_' +str(best_quad_f1) + '.pt'
    # model_path = args.model_dir + 'laptop_0.41293752769162606.pt'
    model_path = args.model_dir + 'restaurant_0.5885057471264368.pt'

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
    hyper_result,opinion_imp_list = eval(Bert, step_1_model, step2_forward_model, step2_reverse_model,step_3_category, testset, args)
    return hyper_result



# 定义搜索空间
space = {
    'drop_out': hp.uniform('drop_out', 0.0, 0.5),
    'task_learning_rate': hp.loguniform('task_learning_rate', -8, -3),
    'train_batch_size': hp.choice('train_batch_size', [4, 8, 16]),
    'kl_loss_weight': hp.uniform('kl_loss_weight', 0.0, 1.0),
    'block_num': hp.choice('block_num', [1, 2, 3])
}


# 定义评估函数
def evaluate(args):
    # 在这里使用传入的超参数进行模型训练和评估
    # 返回一个评估指标（例如，验证集上的准确率或损失）
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument('--model_dir', type=str, default="savemodels/", help='model path prefix')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--init_model", default="pretrained_models/bert-base-uncased", type=str, required=False,
                        help="Initial model.")
    parser.add_argument("--init_vocab", default="pretrained_models/bert-base-uncased", type=str, required=False,
                        help="Initial vocab.")

    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--drop_out", type=int, default=0.1, help="")
    parser.add_argument("--max_span_length", type=int, default=12, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200, help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-6)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--muti_gpu", default=True)
    parser.add_argument('--epochs', type=int, default=200, help='training epoch number')
    parser.add_argument("--train_batch_size", default=1, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=2023, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset_path", default="./datasets/ASTE-Data-V2-EMNLP2020/",
                        choices=["./datasets/BIO_form/", "./datasets/ASTE-Data-V2-EMNLP2020/",
                                 "./datasets/Restaurant-ACOS/"],
                        help="")
    parser.add_argument("--dataset", default="restaurant", type=str, choices=["restaurant", "laptop"],
                        help="specify the dataset")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # 选择Span的合成方式
    parser.add_argument('--span_generation', type=str, default="Max",
                        choices=["Start_end", "Max", "Average", "CNN", "ATT"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")

    # 是否对相关span添加分离Loss
    parser.add_argument("--kl_loss", default=True)
    parser.add_argument("--kl_loss_weight", type=int, default=1, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')

    parser.add_argument("--binary_weight", type=int, default=6, help="weight of the binary loss")
    # 是否使用测试中的筛选算法
    parser.add_argument('--Filter_Strategy', default=True, help='是否使用筛选算法去除冲突三元组')
    # 已被弃用    相关Span注意力
    parser.add_argument("--related_span_underline", default=False)
    parser.add_argument("--related_span_block_num", type=int, default=1,
                        help="number of block in related span attention")

    # 选择Cross Attention中ATT块的个数
    parser.add_argument("--block_num", type=int, default=1, help="number of block")
    parser.add_argument("--output_path", default='triples.json')
    # 按照句子的顺序输入排序
    parser.add_argument("--order_input", default=True, help="")
    '''随机化输入span排序'''
    parser.add_argument("--random_shuffle", type=int, default=0, help="")
    # 验证模型复杂度
    parser.add_argument("--model_para_test", default=False)
    # 使用Warm up快速收敛
    parser.add_argument('--whether_warm_up', default=False)
    parser.add_argument('--warm_up', type=float, default=0.1)
    args = parser.parse_args()

    # 在这里添加你的模型训练和评估代码
    result = train(args)
    # 这里的示例评估指标为随机生成的，你需要根据你的任务替换为实际指标
    return result[2]


# #使用TPE算法进行超参数搜索
# best = fmin(fn=evaluate, space=space, algo=tpe.suggest, max_evals=100)
#
# # 打印搜索得到的最佳超参数组合
# print("Best hyperparameters:")
# print(best)


def load_with_single_gpu(model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    final_state = {}
    for i in state_dict:
        for k, v in state_dict[i].items():
            name = k[7:]
            new_state_dict[name] = v
        final_state[i] = new_state_dict
        new_state_dict = OrderedDict()
    return  final_state

def main():
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument('--model_dir', type=str, default="savemodels/", help='model path prefix')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--init_model", default="pretrained_models/bert-base-uncased", type=str, required=False,help="Initial model.")
    parser.add_argument("--init_vocab", default="pretrained_models/bert-base-uncased", type=str, required=False,help="Initial vocab.")

    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=True, action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=120, type=int,help="The maximum total input sequence length after WordPiece tokenization. "
                                                                       "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--drop_out", type=int, default=0.5, help="")
    parser.add_argument("--max_span_length", type=int, default=12, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,help="")
    parser.add_argument("--task_learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--muti_gpu", default=True)
    parser.add_argument('--epochs', type=int, default=300, help='training epoch number')
    parser.add_argument("--train_batch_size", default=4, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=2023, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset", default="restaurant", type=str, choices=["restaurant", "laptop"],help="specify the dataset")
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # 选择Span的合成方式
    parser.add_argument('--span_generation', type=str, default="Max", choices=["Start_end", "Max", "Average", "CNN", "ATT","Start_end_minus_plus"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")


    # 是否对相关span添加分离Loss
    parser.add_argument("--kl_loss", default=True)
    parser.add_argument("--kl_loss_weight", type=int, default=1, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')
    parser.add_argument("--binary_weight", type=int, default=4, help="weight of the binary loss")

    parser.add_argument("--temp", type=int, default=0.1, help="temperature")
    parser.add_argument("--con_weight", type=int, default=0.1, help="con_weight")
    # 是否使用测试中的筛选算法
    parser.add_argument('--Filter_Strategy',  default=True, help='是否使用筛选算法去除冲突三元组')
    # 已被弃用    相关Span注意力
    parser.add_argument("--related_span_underline", default=False)
    parser.add_argument("--related_span_block_num", type=int, default=1, help="number of block in related span attention")

    # 选择Cross Attention中ATT块的个数
    parser.add_argument("--block_num", type=int, default=1, help="number of block")
    parser.add_argument("--output_path", default='triples.json')
    #按照句子的顺序输入排序
    parser.add_argument("--order_input", default=True, help="")
    '''随机化输入span排序'''
    parser.add_argument("--random_shuffle", type=int, default=0, help="")
    # 验证模型复杂度
    parser.add_argument("--model_para_test", default=False)
    # 使用Warm up快速收敛
    parser.add_argument('--whether_warm_up', default=False)
    parser.add_argument('--warm_up', type=float, default=0.1)
    args = parser.parse_args()

    for k,v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    train(args)


if __name__ == "__main__":
    '''
    # 设置显存阈值
    memory_threshold = 4100  # GB

    while True:
        # 获取所有可用的GPU信息
        gpu_info = GPUtil.getGPUs()

        # 初始化一个列表来存储可用的GPU索引
        available_gpu_indices = []

        # 遍历每张显卡
        for i, gpu in enumerate(gpu_info):
            logger.info(f"GPU {i + 1} - 显存剩余: {gpu.memoryFree} MB")

            # 检查显存是否大于阈值
            if gpu.memoryFree > memory_threshold:
                available_gpu_indices.append(i)

        if available_gpu_indices:
            # 将可用的GPU设备索引设置为CUDA可见设备
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpu_indices))
            logger.info(f"已选择可用的GPU设备: {os.environ['CUDA_VISIBLE_DEVICES']}")
            break  # 退出循环并继续执行任务

        # 如果没有满足条件的显卡，等待一段时间后重新检查
        time.sleep(60)  # 每隔60秒重新检查一次
    '''
    try:
        main()

    except KeyboardInterrupt:
        logger.info("keyboard break")
