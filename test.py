import torch.nn as nn
import os
import argparse
import tqdm
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from models.data_BIO_loader import load_data, DataTterator,DataTterator2
from models.model import stage_2_features_generation, Step_1, Step_2_forward, Step_2_reverse, Loss
from models.Metric import Metric
from models.eval_features import unbatch_data
from log import logger
from thop import profile, clever_format
import time
import torch
import numpy as np
import random
import json
from transformers import BertTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
validity2id = {'none': 0, 'positive': 1, 'negative': 1, 'neutral': 1}
sentiment2id = {'none': 3, 'positive': 2, 'negative': 0, 'neutral': 1}
categories = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
category2id={}
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
    parser.add_argument("--max_span_length", type=int, default=8, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--muti_gpu", default=True)
    parser.add_argument('--epochs', type=int, default=130, help='training epoch number')
    parser.add_argument("--train_batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=2022, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset_path", default="./datasets/ASTE-Data-V2-EMNLP2020/",
                        choices=["./datasets/BIO_form/", "./datasets/ASTE-Data-V2-EMNLP2020/"],
                        help="")
    parser.add_argument("--dataset", default="lap14", type=str, choices=["lap14", "res14", "res15", "res16"],
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

    # for k,v in sorted(vars(args).items()):
    #     logger.info(str(k) + '=' + str(v))
    train(args)

def load_data1(args, path, if_train=False):

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if if_train:
        random.seed(args.RANDOM_SEED)
        random.shuffle(lines)

    list_instance_batch={}
    instances = load_data_instances_txt(lines)
    data_instances, aspect_num, num_opinion = convert_examples_to_features(args, train_instances=instances,
                                                                           max_span_length=args.max_span_length)
    list_instance_batch = []
    for i in range(0, len(data_instances), args.train_batch_size):
        list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    return list_instance_batch

def convert_examples_to_features(args, train_instances, max_span_length=8):

    features = []
    num_aspect = 0
    num_quad = 0
    num_opinion = 0
    differ_opinion_senitment_num = 0

    id2category = {i + 1: ch for i, ch in enumerate(categories)}
    id2category[0]=0
    category2id = {v: k for k, v in id2category.items()}
    for ex_index, example in enumerate(train_instances):
        sample = {'id': example.id}
        sample['tokens'] = example.text_a.split(' ')
        sample['text_length'] = len(sample['tokens'])
        sample['quad'] = example.all_label
        sample['sentence'] = example.text_a
        aspect = {}
        opinion = {}
        category = {}
        opinion_reverse = {}
        aspect_reverse  = {}

        differ_opinion_sentiment = False

        for quad_name in sample['quad']:
            aspect_span, opinion_span, categorystr,sentiment = tuple(sample['quad'][quad_name][0]), tuple(
                sample['quad'][quad_name][1]), sample['quad'][quad_name][2],sample['quad'][quad_name][3]
            num_quad += 1
            if aspect_span not in aspect:
                aspect[aspect_span] = sentiment
                category[aspect_span] = categorystr
                opinion[aspect_span] = [(opinion_span, sentiment)]
            else:
                # assert aspect[aspect_span] == sentiment
                # if aspect[aspect_span] != sentiment:
                #     differ_opinion_sentiment = True
                #     print(sample+'\n'+aspect[aspect_span]+sentiment)
                opinion[aspect_span].append((opinion_span, sentiment))

            if opinion_span not in opinion_reverse:
                opinion_reverse[opinion_span] = sentiment
                aspect_reverse[opinion_span] = [(aspect_span, sentiment)]
            else:
                if opinion_reverse[opinion_span] != sentiment:
                    differ_opinion_sentiment = True
                else:
                    aspect_reverse[opinion_span].append((aspect_span, sentiment))
        if differ_opinion_sentiment:
            differ_opinion_senitment_num += 1
            print(ex_index, '单意见词多极性')
            continue

        num_aspect += len(aspect)
        num_opinion += len(opinion)

        # if len(aspect) != example.aspect_num:
        #     print('有不同三元组使用重复了aspect:', example.id)

        spans = []
        span_tokens = []

        spans_aspect_label = []
        spans_aspect2opinion_label =[]
        spans_aspect2category_label = []
        spans_opinion_label = []

        reverse_opinion_label = []
        reverse_opinion2aspect_label = []
        reverse_aspect_label = []

        if args.order_input:
            for i in range(max_span_length):
                if sample['text_length'] < i:
                    continue
                for j in range(sample['text_length'] - i):
                    # 所有可能的span（0,0,1）(1,1,1)....
                    spans.append((j, i + j, i + 1))
                    # 当前span的token
                    span_token = ' '.join(sample['tokens'][j:i + j + 1])
                    # 所有的span token
                    span_tokens.append(span_token)
                    if (j, i + j) not in aspect:
                        # 当前span没有情感
                        spans_aspect_label.append(0)
                    else:
                        # spans_aspect_label.append(sentiment2id[aspect[(j, i + j)]])
                        spans_aspect_label.append(validity2id[aspect[(j, i + j)]])
                    if (j, i + j) not in opinion_reverse:
                        # 当前opinion没有情感
                        reverse_opinion_label.append(0)
                    else:
                        # reverse_opinion_label.append(sentiment2id[opinion_reverse[(j, i + j)]])
                        # 对应到标签 添加情感
                        reverse_opinion_label.append(validity2id[opinion_reverse[(j, i + j)]])
                    if (j , i + j) not in category:
                        spans_aspect2category_label.append(0)
                    else:
                        spans_aspect2category_label.append(category2id[category[(j,i+j)]])

        else:
            for i in range(sample['text_length']):
                for j in range(i, min(sample['text_length'], i + max_span_length)):
                    spans.append((i, j, j - i + 1))
                    # sample['spans'].append((i, j, j-i+1))
                    span_token = ' '.join(sample['tokens'][i:j + 1])
                    # sample['span tokens'].append(span_tokens)
                    span_tokens.append(span_token)
                    if (i, j) not in aspect:
                        spans_aspect_label.append(0)
                    else:
                        # spans_aspect_label.append(sentiment2id[aspect[(i, j)]])
                        spans_aspect_label.append(validity2id[aspect[(i, j)]])
                    if (i, j) not in opinion_reverse:
                        reverse_opinion_label.append(0)
                    else:
                        # reverse_opinion_label.append(sentiment2id[opinion_reverse[(i, j)]])
                        reverse_opinion_label.append(validity2id[opinion_reverse[(i, j)]])


        assert len(span_tokens) == len(spans)
        for key_aspect in opinion:
            opinion_list = []
            sentiment_opinion = []
            spans_aspect2opinion_label.append(key_aspect)
            for opinion_span_2_aspect in opinion[key_aspect]:
                opinion_list.append(opinion_span_2_aspect[0])
                sentiment_opinion.append(opinion_span_2_aspect[1])
            # assert len(set(sentiment_opinion)) == 1
            opinion_label2triple = []
            for i in spans:
                if (i[0], i[1]) not in opinion_list:
                    opinion_label2triple.append(3)
                else:
                    opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])
            spans_opinion_label.append(opinion_label2triple)

        for opinion_key in aspect_reverse:
            aspect_list = []
            sentiment_aspect = []
            reverse_opinion2aspect_label.append(opinion_key)
            for aspect_span_2_opinion in aspect_reverse[opinion_key]:
                aspect_list.append(aspect_span_2_opinion[0])
                sentiment_aspect.append(aspect_span_2_opinion[1])
            assert len(set(sentiment_aspect)) == 1
            aspect_label2triple = []
            for i in spans:
                if (i[0], i[1]) not in aspect_list:
                    aspect_label2triple.append(3)
                else:
                    aspect_label2triple.append(sentiment2id[sentiment_aspect[0]])
            reverse_aspect_label.append(aspect_label2triple)

        sample['aspect_num'] = len(spans_opinion_label)
        sample['spans_aspect2opinion_label'] = spans_aspect2opinion_label
        sample['reverse_opinion_num'] = len(reverse_aspect_label)
        sample['reverse_opinion2aspect_label'] = reverse_opinion2aspect_label
        sample['spans_aspect2category_label'] = spans_aspect2category_label

        if args.random_shuffle != 0:
            np.random.seed(args.random_shuffle)
            shuffle_ix = np.random.permutation(np.arange(len(spans)))
            spans_np = np.array(spans)[shuffle_ix]
            span_tokens_np = np.array(span_tokens)[shuffle_ix]
            '''双向同顺序打乱'''
            spans_aspect_label_np = np.array(spans_aspect_label)[shuffle_ix]
            reverse_opinion_label_np = np.array(reverse_opinion_label)[shuffle_ix]
            spans_opinion_label_shuffle = []
            for spans_opinion_label_split in spans_opinion_label:
                spans_opinion_label_split_np = np.array(spans_opinion_label_split)[shuffle_ix]
                spans_opinion_label_shuffle.append(spans_opinion_label_split_np.tolist())
            spans_opinion_label = spans_opinion_label_shuffle
            reverse_aspect_label_shuffle = []
            for reverse_aspect_label_split in reverse_aspect_label:
                reverse_aspect_label_split_np = np.array(reverse_aspect_label_split)[shuffle_ix]
                reverse_aspect_label_shuffle.append(reverse_aspect_label_split_np.tolist())
            reverse_aspect_label = reverse_aspect_label_shuffle
            spans, span_tokens, spans_aspect_label, reverse_opinion_label  = spans_np.tolist(), span_tokens_np.tolist(),\
                                                                             spans_aspect_label_np.tolist(), reverse_opinion_label_np.tolist()
        related_spans = np.zeros((len(spans), len(spans)), dtype=int)
        for i in range(len(span_tokens)):
            span_token = span_tokens[i].split(' ')
            # for j in range(i, len(span_tokens)):
            for j in range(len(span_tokens)):
                differ_span_token = span_tokens[j].split(' ')
                if set(span_token) & set(differ_span_token) == set():
                    related_spans[i, j] = 0
                else:
                    related_spans[i, j] = 1

        sample['related_span_array'] = related_spans
        sample['spans'], sample['span tokens'], sample['spans_aspect_label'], sample[
            'spans_opinion_label'] = spans, span_tokens, spans_aspect_label, spans_opinion_label
        sample['reverse_opinion_label'], sample['reverse_aspect_label'] = reverse_opinion_label, reverse_aspect_label
        features.append(sample)
    return features, num_aspect, num_opinion
class InputExample(object):
    def __init__(self, id, text_a, aspect_num, triple_num, all_label=None, text_b=None):
        """Build a InputExample"""
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.all_label = all_label
        self.aspect_num = aspect_num
        self.triple_num = triple_num

def load_data_instances_txt(lines):
    id2sentiment = {'0': 'negative', '1': 'neutral', '2': 'positive'}

    instances = list()
    quad_num = 0
    aspects_num = 0
    for ex_index, line in enumerate(lines):
        id = str(ex_index)  # id
        line = line.strip()
        line = line.split('\t')
        sentence = line[0].split()  # sentence
        # raw_pairs = eval(line[1])  # triplets
        raw_pairs = line[1:]
        if ex_index==3:
            print(sentence,raw_pairs)
        quad_dict = {}
        aspect_num = 0
        for quad in raw_pairs:
            quad=quad.split(' ')
            raw_aspect = quad[0]
            raw_aspect_tuple = list(map(int, raw_aspect.split(',')))
            raw_category = quad[1]
            sentiment = id2sentiment[quad[2]]
            raw_opinion = quad[3]
            raw_opinion_tuple = list(map(int, raw_opinion.split(',')))
            # print(raw_aspect,raw_opinion,raw_category,sentiment)
            aspect_word = ' '.join(sentence[raw_aspect_tuple[0]: raw_aspect_tuple[1]])
            raw_aspect = [raw_aspect_tuple[0], raw_aspect_tuple[1]-1]
            aspect_label = {}
            aspect_label[aspect_word] = raw_aspect
            aspect_num += len(aspect_label)

            opinion_word = ' '.join(sentence[raw_opinion_tuple[0]: raw_opinion_tuple[1]])
            raw_opinion = [raw_opinion_tuple[0], raw_opinion_tuple[1] - 1]
            opinion_label = {}
            opinion_label[opinion_word] = raw_opinion
            # print(opinion_word, raw_opinion,opinion_label)
            # print('$$$$$$$$$$$$$$$$$$$')


            word = str(aspect_word) + '|' + str(opinion_word)
            if word not in quad_dict:
                quad_dict[word] = []
                quad_dict[word] = ([raw_aspect[0], raw_aspect[-1]], [raw_opinion[0], raw_opinion[-1]], raw_category ,sentiment)
            else:
                print('单句' + id + '中三元组重复出现！')
        examples = InputExample(id=id, text_a=line[0], text_b=None, all_label=quad_dict, aspect_num=aspect_num,
                                    triple_num=len(quad_dict))

        instances.append(examples)
        quad_num += quad_num
        aspects_num += aspect_num


    return instances
def train(args):
    torch.cuda.current_device()
    torch.cuda._initialized = True
    train_path = "./datasets/Restaurant-ACOS/rest16_quad_train.tsv"
    # train_path = './datasets/ASTE-Data-V2-EMNLP2020/lap14/train_triplets.txt'
    train_datasets = load_data1(args, train_path, if_train=False)
    train_set = DataTterator2(train_datasets, args)
    for i in range(args.epochs):
        logger.info(('Epoch:{}'.format(i)))
        for j in tqdm.trange(train_set.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length,spans_category_label_tensor \
                = train_set.get_batch(j)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("keyboard break")
