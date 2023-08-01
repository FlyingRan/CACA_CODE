import os
import argparse
import tqdm
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from models.data_BIO_loader import load_data, load_data1,DataTterator,DataTterator2
from models.model import stage_2_features_generation, Step_1, Step_2_forward, Step_2_reverse, Loss,Step_3_categories
from models.Metric import Metric
from models.eval_features import unbatch_data
from log import logger
from thop import profile, clever_format
import time
import sys
import codecs
import numpy as np
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sentiment2id = {'none': 3, 'positive': 2, 'negative': 0, 'neutral': 1}

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
        is_aspect = False
        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
            reverse_aspect_label_tensor, related_spans_tensor, sentence_length,spans_category_label_tensor = dataset.get_batch(j)



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

            reverse_pred_stage1_logits = torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2)
            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)
            reverse_pred_stage1_logits = torch.where(spans_mask_tensor == 1, reverse_pred_stage1_logits,
                                             torch.tensor(0).type_as(reverse_pred_stage1_logits))

            pred_aspect_logits[0][0] = pred_imp_aspect[0]
            pred_aspect_logits[0][len(pred_aspect_logits[0]) - 1] = 0

            reverse_pred_stage1_logits[0][len(reverse_pred_stage1_logits[0]) - 1] = pred_imp_opinion[0]
            reverse_pred_stage1_logits[0][0] = 0
            '''真实结果合成'''
            gold_instances.append(dataset.instances[j])
            result = []

            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] !=0\
                    and torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] != 0:
                opinion_span = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                           torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                aspect_span = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                          torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                pred_pair = torch.tensor(
                    [[aspect_span[i][0][1], opinion_span[j][0][1]]
                     for i in range(len(aspect_span)) for j in range(len(opinion_span))])
                aspect_rep = []
                opinion_rep = []
                for aspect_idx in aspect_span:
                    aspect_rep.append(spans_embedding[aspect_idx[0][0], aspect_idx[0][1]])
                aspect_rep = torch.stack(aspect_rep)
                for opinion_idx in opinion_span:
                    opinion_rep.append(spans_embedding[opinion_idx[0][0], opinion_idx[0][1]])
                opinion_rep = torch.stack(opinion_rep)
                # 使用广播计算笛卡尔积并拼接
                expanded_aspect = aspect_rep.unsqueeze(1)
                expanded_opinion = opinion_rep.unsqueeze(0)
                expanded_aspect = expanded_aspect.repeat(1, opinion_rep.size(0), 1)
                expanded_opinion = expanded_opinion.repeat(aspect_rep.size(0), 1, 1)
                cartesian_product = torch.cat((expanded_aspect, expanded_opinion), dim=2)
                # 转换形状
                input_rep  = cartesian_product.reshape(aspect_rep.size(0) * opinion_rep.size(0), 768 * 2)
                category_logits,_ = step_3_category(spans_embedding,bert_spans_tensor,input_rep)
                pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                result = []
                for i,pair in enumerate(pred_pair):
                    a_o_c_result = pair.tolist()
                    a_o_c_result.append(int(pred_category_logits[i]))
                    result.append(a_o_c_result)
                category_result.append(result)
            else:
                category_result.append([])

            '''双方向预测'''
            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:

                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                forward_stage2_pred_opinion_sentiment_logit.append(
                    torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
                # forward_pred_category_logit.append(pred_category_logits)


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
                # _,all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, all_span_mask = stage_2_features_generation(
                #     bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                #     spans_embedding, pred_span_aspect_tensor)
                is_aspect = True
                _, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, all_span_mask = stage_2_features_generation(
                    bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    forward_embedding,pred_span_aspect_tensor,is_aspect)

                opinion_class_logits, opinion_attention = step_2_forward(all_spans_embedding, all_span_mask,
                                                                         all_span_aspect_tensor)
                #step1. 找到所有的opinion_index及其表征
                #step2. 遍历aspect个数，逐个拼接opinion表征，最后输入到step3_category中
                #step3. 添加到result
                '''
                output_values, output_indices = torch.max(opinion_class_logits, dim=-1)
                opinion_list = []
                for row in output_indices:
                    non_three_indices_row = torch.nonzero(row != 3, as_tuple=False)
                    opinion_list.append(non_three_indices_row)

                for i, row_indices in enumerate(opinion_list):
                    # print(f"第{i + 1}行不等于3的索引位置：")
                    if row_indices.size(0) == 0:
                        continue
                    opinion_rep = []

                    for index in row_indices:
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        opinion_rep.append(spans_embedding[0][index.item()])
                    opinion_rep = torch.stack(opinion_rep)
                    aspect_rep = all_span_aspect_tensor[i][0].clone()
                    aspect_repx3 = aspect_rep.expand(len(row_indices),-1).to(args.device)
                    final_rep = torch.cat((aspect_repx3,opinion_rep),dim=1)
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j,index in enumerate(row_indices):
                        result.append([int(pred_aspect_spans[i][0][1]),int(index),int(pred_category_logits[j])])

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
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                   ## if int(reverse_pred_opinion_span.squeeze()[1]) == 0:
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
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(
                    bert_features.last_hidden_state,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    reverse_embedding,
                    reverse_span_opinion_tensor,
                    is_aspect)

                reverse_aspect_class_logits, reverse_aspect_attention = step_2_reverse(reverse_spans_embedding,
                                                                                reverse_span_mask,
                                                                                all_reverse_opinion_tensor)
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
                    for index in row_indices:
                        # print(index.item())
                        # print(spans_embedding[0][index.item()])
                        aspect_rep.append(spans_embedding[0][index.item()])
                    aspect_rep = torch.stack(aspect_rep)
                    opinion_rep = all_reverse_opinion_tensor[i][0].clone()
                    opinion_repx3 = opinion_rep.expand(len(row_indices), -1).to(args.device)
                    final_rep = torch.cat((aspect_rep, opinion_repx3), dim=1)
                    category_logits, _ = step_3_category(spans_embedding, bert_spans_tensor, final_rep)
                    pred_category_logits = torch.argmax(F.softmax(category_logits, dim=1), dim=1)
                    for j, index in enumerate(row_indices):
                        result.append([int(index), int(reverse_pred_opinion_spans[i][0][1]), int(pred_category_logits[j])])
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
                             forward_stage2_pred_opinion_sentiment_logit,category_result)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit)
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
        logger.info('quad precision: {}\tquad recall: {:.8f}\tquad f1: {:.8f}'.format(quad_result[0],
                                                                                            quad_result[1],
                                                                                            quad_result[2]))

    bert_model.train()
    step_1_model.train()
    step_2_forward.train()
    step_2_reverse.train()
    step_3_category.train()
    return quad_result


def train(args):
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

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
        train_datasets = load_data1(args, train_path, if_train=True)
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
        if args.whether_warm_up:
            training_steps = args.epochs * devset.batch_count
            warmup_steps = int(training_steps * args.warm_up)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)

        tot_loss = 0
        tot_kl_loss = 0
        best_aspect_f1, best_opinion_f1, best_APCE_f1, best_pairs_f1, best_quad_f1,best_quad_precision,best_quad_recall = 0,0,0,0,0,0,0
        best_quad_epoch= 0
        for i in range(args.epochs):
            logger.info(('Epoch:{}'.format(i)))

            for j in tqdm.trange(trainset.batch_count):
            # for j in range(trainset.batch_count):
                if j == 1:
                    start = time.time()
                # if j == 1:
                #     print()
                optimizer.zero_grad()
                # token_tensor:一个长为 batch_size 的张量，其中每个元素是一个长度为 max_seq_length(100) 的整数列表
                # attention_mask : 与 tokens_tensor 大小相同的张量，其中每个元素的值为 0 或 1，用于指示哪些 token 是真正的输入（即不是填充项）
                # bert_spans_tensor: 一个大小为 (batch_size, max_num_spans, 3) 的张量，其中每个元素是一个长度为 3的整数列表
                # spans_mask_tensor: 一个大小为 (batch_size, max_num_spans) 的张量，其中每个元素的值为 0 或 1，用于指示哪些语言单元在当前样本中是存在的。
                # spans_aspect_tensor: 一个大小为 (三元组数, 3) 的张量，3=第几条句子+两个aspect跨度
                # spans_opinion_label_tensor: 一个大小为 (batch_size, max_num_spans) 的张量，其中每个元素是一个整数，表示对应语言单元的情感标签。
                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length,spans_category_label_tensor = trainset.get_batch(j)

                bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)
                # class_logits_aspect（shape=[batch_size, 2]）
                # class_logits_opinion（shape=[batch_size, 2]）
                # aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                # cnn_spans_mask_tensor= step_1_model(
                #                                                                       bert_output.last_hidden_state,
                #                                                                       attention_mask,
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

                category_logits,category_label = step_3_category(spans_embedding,bert_spans_tensor,sentence_length[0][3],sentence_length[0][4][0])
                is_aspect = True
                '''Batch更新'''
                # 21X420                     21 1 768           21 100 768           21 100        21 420 768    21 420
                # 从输入的 BERT 特征、注意力掩码、span 序列和 span 掩码中生成经过处理后的特征
                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                all_spans_embedding, all_span_mask = stage_2_features_generation(bert_output.last_hidden_state,
                                                                             attention_mask, bert_spans_tensor,
                                                                             spans_mask_tensor, forward_embedding,
                                                                             spans_aspect_tensor,
                                                                                is_aspect,
                                                                             spans_opinion_label_tensor,
                                                                                 )
                is_aspect = False
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_output.last_hidden_state,
                                                                                     attention_mask, bert_spans_tensor,
                                                                                     spans_mask_tensor, reverse_embedding,
                                                                                     reverse_opinion_tensor,
                                                                                        is_aspect,
                                                                                     reverse_aspect_label_tensor
                                                                                         )

                # aspect->opinion
                step_2_opinion_class_logits, opinion_attention = step2_forward_model(all_spans_embedding, all_span_mask, all_span_aspect_tensor)
                # opinion->aspect
                step_2_aspect_class_logits, aspect_attention = step2_reverse_model(reverse_spans_embedding,
                    reverse_span_mask, all_reverse_opinion_tensor)
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
            if i==10:
                print()
            print('Evaluating, please wait')
            # aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, step_1_model,
            #                                                                                step2_forward_model,
            #                                                                                step2_reverse_model,
            #                                                                                devset, args)
            quad_result = eval(Bert, step_1_model, step2_forward_model, step2_reverse_model,step_3_category, testset, args)
            print('Evaluating complete')
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

            if quad_result[2] > best_quad_f1 and quad_result[2] > 0.35:
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
                'best triple epoch: {}\tbest triple precision: {:.8f}\tbest triple recall: {:.8f}\tbest triple f1: {:.8f}'.
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
            'best triple epoch: {}\tbest triple precision: {:.8f}\tbest triple recall: {:.8f}\tbest triple f1: {:.8f}'.
            format(best_quad_epoch, best_quad_precision, best_quad_recall, best_quad_f1))

    logger.info("Features build completed")
    logger.info("Evaluation on testset:")

    model_path = args.model_dir + args.dataset+'_'+str(best_quad_f1) + '.pt'
    #model_path = args.model_dir + 'restaurant_0.45739910313901344.pt'
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
    eval(Bert, step_1_model, step2_forward_model, step2_reverse_model,step_3_category, testset, args)

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
    parser.add_argument("--max_seq_length", default=100, type=int,help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--drop_out", type=int, default=0.1, help="")
    parser.add_argument("--max_span_length", type=int, default=12, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--muti_gpu", default=True)
    parser.add_argument('--epochs', type=int, default=150, help='training epoch number')
    parser.add_argument("--train_batch_size", default=1, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=2022, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset_path", default="./datasets/ASTE-Data-V2-EMNLP2020/",
                        choices=["./datasets/BIO_form/", "./datasets/ASTE-Data-V2-EMNLP2020/","./datasets/Restaurant-ACOS/"],
                        help="")
    parser.add_argument("--dataset", default="restaurant", type=str, choices=["restaurant", "laptop"],
                        help="specify the dataset")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # 选择Span的合成方式
    parser.add_argument('--span_generation', type=str, default="Max", choices=["Start_end", "Max", "Average", "CNN", "ATT"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")

    # 是否对相关span添加分离Loss
    parser.add_argument("--kl_loss", default=True)
    parser.add_argument("--kl_loss_weight", type=int, default=0.5, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')
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
    try:
        main()

    except KeyboardInterrupt:
        logger.info("keyboard break")
