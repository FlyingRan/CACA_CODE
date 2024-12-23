
import torch
import numpy as np
import random
import json
from transformers import BertTokenizer,AutoTokenizer
from test import filter_invalid_spans,is_valid_span
import torch.nn.functional as F
validity2id = {'none': 0, 'positive': 1, 'negative': 1, 'neutral': 1}

sentiment2id = {'none': 3, 'positive': 2, 'negative': 0, 'neutral': 1}


def get_categories(args):
    if args.dataset == "restaurant":
        categories = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
    elif args.dataset == "laptop":
        categories = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE',
                      'CPU#OPERATION_PERFORMANCE',
                      'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE',
                      'POWER_SUPPLY#CONNECTIVITY', 'SOFTWARE#USABILITY',
                      'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY',
                      'FANS&COOLING#DESIGN_FEATURES',
                      'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY',
                      'POWER_SUPPLY#GENERAL', 'PORTS#QUALITY',
                      'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY',
                      'MOUSE#GENERAL', 'KEYBOARD#MISCELLANEOUS',
                      'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS',
                      'SOFTWARE#PRICE', 'FANS&COOLING#OPERATION_PERFORMANCE',
                      'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL',
                      'DISPLAY#OPERATION_PERFORMANCE',
                      'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY',
                      'KEYBOARD#PRICE', 'SUPPORT#OPERATION_PERFORMANCE',
                      'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY',
                      'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES',
                      'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL',
                      'PORTS#USABILITY', 'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY',
                      'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY',
                      'HARDWARE#DESIGN_FEATURES', 'MEMORY#USABILITY',
                      'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY',
                      'OS#PRICE', 'SUPPORT#QUALITY',
                      'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL',
                      'COMPANY#OPERATION_PERFORMANCE',
                      'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES',
                      'Out_Of_Scope#OPERATION_PERFORMANCE',
                      'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY',
                      'DISPLAY#USABILITY', 'POWER_SUPPLY#QUALITY',
                      'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY',
                      'HARDWARE#GENERAL', 'COMPANY#PRICE',
                      'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE',
                      'SOFTWARE#PORTABILITY', 'HARD_DISC#OPERATION_PERFORMANCE',
                      'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES',
                      'OS#OPERATION_PERFORMANCE', 'OS#USABILITY',
                      'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE',
                      'LAPTOP#PRICE', 'OS#GENERAL', 'HARDWARE#PRICE',
                      'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY',
                      'FANS&COOLING#QUALITY', 'BATTERY#OPERATION_PERFORMANCE',
                      'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE',
                      'KEYBOARD#GENERAL', 'SOFTWARE#QUALITY',
                      'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE',
                      'WARRANTY#QUALITY', 'HARD_DISC#QUALITY',
                      'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
    elif args.dataset == "phone":
        categories = [
            'PRODUCT.ACCESSORIES#PHONE.CASES', 'PRODUCT.QUALITY#DUSTPROOF',
            'BUYER.ATTITUDE#REPURCHASE.AND.CHURN.TENDENCY', 'PRODUCT.QUALITY#GENERAL',
            'PRODUCT.QUALITY#GENUINE.PRODUCT', 'INTELLIGENT.ASSISTANT#WAKE-UP.FUNCTION',
            'AUDIO/SOUND#VOLUME.AND.SPEAKER', 'SELLER.SERVICE#INVENTORY', 'APPEARANCE.DESIGN#WORKMANSHIP.AND.TEXTURE',
            'SIGNAL#WIFI.SIGNAL', 'PRODUCT.CONFIGURATION#OPERATING.MEMORY', 'PRODUCT.QUALITY#WATER.RESISTANT',
            'BATTERY/LONGEVITY#STANDBY.TIME', 'PRODUCT.PACKAGING#GENERAL', 'PRODUCT.QUALITY#CLEANLINESS',
            'BATTERY/LONGEVITY#BATTERY.LIFE', 'PRODUCT.PACKAGING#PACKAGING.MATERIALS',
            'SELLER.SERVICE#SELLER.EXPERTISE', 'OVERALL#OVERALL', 'PRICE#PRICE',
            'INTELLIGENT.ASSISTANT#INTELLIGENT.ASSISTANT.GENERAL', 'SYSTEM#SYSTEM.GENERAL',
            'PRODUCT.ACCESSORIES#HEADPHONES', 'APPEARANCE.DESIGN#EXTERIOR.DESIGN.MATERIAL',
            'SELLER.SERVICE#TIMELINESS.OF.SELLER.SERVICE', 'PRODUCT.QUALITY#FALL.PROTECTION',
            'APPEARANCE.DESIGN#FUSELAGE.SIZE', 'BUYER.ATTITUDE#LOYALTY', 'SYSTEM#OPERATION.SMOOTHNESS',
            'BATTERY/LONGEVITY#CHARGING.SPEED', 'BATTERY/LONGEVITY#GENERAL', 'SHOOTING.FUNCTIONS#GENERAL',
            'SYSTEM#LOCK.SCREEN.DESIGN', 'PERFORMANCE#GENERAL', 'APPEARANCE.DESIGN#COLOR', 'APPEARANCE.DESIGN#WEIGHT',
            'BUYER.ATTITUDE#RECOMMENDABLE', 'SIGNAL#SIGNAL.OF.MOBILE.NETWORK', 'EASE.OF.USE#AUDIENCE.GROUPS',
            'PRODUCT.PACKAGING#PACKAGING.GRADE', 'SYSTEM#NFC', 'SMART.CONNECT#POSITIONING.AND.GPS',
            'BRANDING/MARKETING#PROMOTIONAL.GIVEAWAYS', 'PRODUCT.ACCESSORIES#CHARGING.CABLE', 'SELLER.SERVICE#ATTITUDE',
            'BUYER.ATTITUDE#SHOPPING.WILLINGNESS', 'AUDIO/SOUND#TONE.QUALITY', 'SYSTEM#APPLICATION',
            'SHOOTING.FUNCTIONS#PIXEL', 'SECURITY#SCREEN.UNLOCK', 'AFTER-SALES.SERVICE#EXCHANGE/WARRANTY/RETURN',
            'BATTERY/LONGEVITY#POWER.CONSUMPTION.SPEED', 'BUYER.ATTITUDE#SHOPPING.EXPERIENCES',
            'BATTERY/LONGEVITY#CHARGING.METHOD', 'PRICE#VALUE.FOR.MONEY',
            'PRODUCT.PACKAGING#COMPLETENESS.OF.ACCESSORIES', 'PRODUCT.ACCESSORIES#CHARGER', 'PERFORMANCE#RUNNING.SPEED',
            'SCREEN#SIZE', 'SCREEN#CLARITY', 'LOGISTICS#GENERAL', 'PRODUCT.PACKAGING#INSTRUCTION.MANUAL',
            'SCREEN#SCREEN-TO-BODY.RATIO', 'LOGISTICS#SHIPPING.FEE', 'KEY.DESIGN#GENERAL',
            'PRODUCT.ACCESSORIES#CELL.PHONE.FILM', 'APPEARANCE.DESIGN#THICKNESS', 'LOGISTICS#SPEED',
            'APPEARANCE.DESIGN#AESTHETICS.GENERAL', 'APPEARANCE.DESIGN#GRIP.FEELING',
            'BATTERY/LONGEVITY#BATTERY.CAPACITY', 'SCREEN#GENERAL', 'SYSTEM#SYSTEM.UPGRADE',
            'SYSTEM#SOFTWARE.COMPATIBILITY', 'SIGNAL#CALL.QUALITY', 'SIGNAL#SIGNAL.GENERAL', 'EASE.OF.USE#EASY.TO.USE',
            'LOGISTICS#LOST.AND.DAMAGED', 'SELLER.SERVICE#SHIPPING', 'CAMERA#FRONT.CAMERA',
            'SYSTEM#UI.INTERFACE.AESTHETICS', 'CAMERA#GENERAL', 'PRODUCT.CONFIGURATION#MEMORY', 'CAMERA#REAR.CAMERA',
            'CAMERA#FILL.LIGHT', 'PRODUCT.CONFIGURATION#CPU', 'PERFORMANCE#HEAT.GENERATION',
            'SMART.CONNECT#BLUETOOTH.CONNECTION'
        ]
    else:
        categories = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL']
    category2id = {}
    id2category = {i: ch for i, ch in enumerate(categories)}
    return categories,id2category

def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_subject_labels(tags):
    '''for BIO tag'''

    label = {}
    subject_span = get_spans(tags)[0]
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    word = ' '.join(sentence[subject_span[0]:subject_span[1] + 1])
    label[word] = subject_span
    return label


def get_object_labels(tags):
    '''for BIO tag'''
    label = {}
    object_spans = get_spans(tags)
    tags = tags.strip().split()
    sentence = []
    for tag in tags:
        sentence.append(tag.strip().split('\\')[0])
    for object_span in object_spans:
        word = ' '.join(sentence[object_span[0]:object_span[1] + 1])
        label[word] = object_span
    return label


class InputExample(object):
    def __init__(self, id, text_a, aspect_num, triple_num, all_label=None, text_b=None):
        """Build a InputExample"""
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.all_label = all_label
        self.aspect_num = aspect_num
        self.triple_num = triple_num


class Instance(object):
    def __init__(self, sentence_pack, args):
        triple_dict = {}
        id = sentence_pack['id']
        aspect_num = 0
        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            sentiment = triple['sentiment']
            subject_label = get_subject_labels(aspect)
            object_label = get_object_labels(opinion)
            objects = list(object_label.keys())
            subject = list(subject_label.keys())[0]
            aspect_num += len(subject_label)
            for i, object in enumerate(objects):
                # 由于数据集的每个triples中aspect只有一个，而opinion可能有多个  需要分开构建
                word = str(subject) + '|' + str(object)
                if word not in triple_dict:
                    triple_dict[word] = []
                triple_dict[word] = (subject_label[subject], object_label[object], sentiment)
        examples = InputExample(id=id, text_a=sentence_pack['sentence'], text_b=None, all_label=triple_dict,
                                aspect_num=aspect_num, triple_num=len(triple_dict))
        self.examples = examples
        self.triple_num = len(triple_dict)
        self.aspect_num = aspect_num


def load_data_instances(sentence_packs, args):
    instances = list()
    triples_num = 0
    aspects_num = 0
    for i, sentence_pack in enumerate(sentence_packs):
        instance = Instance(sentence_pack, args)
        instances.append(instance.examples)
        triples_num += instance.triple_num
        aspects_num += instance.aspect_num
    return instances





def write_json_data(path,dataset):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for ex_index, line in enumerate(lines):
        line = line.strip()
        line = line.split('\t')
        sentence = line[0].split()  # sentence
        # 将数据组织成字典
        data_dict = {"tokens": sentence}
        # 将字典添加到结果列表中
        result.append(data_dict)

    # 写入 JSON 文件
    output_file_path = dataset+ "_output.json"
    with open(output_file_path, "w") as json_file:
        json.dump(result, json_file, indent=2)
def load_data1(args, path, if_train=False):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if if_train:
        random.seed(args.RANDOM_SEED)
        random.shuffle(lines)

    list_instance_batch = {}
    instances = load_data_instances_txt1(lines,args)
    data_instances, aspect_num, num_opinion = convert_examples_to_features1(args, train_instances=instances,
                                                                           max_span_length=args.max_span_length)
    list_instance_batch = []
    for i in range(0, len(data_instances), args.train_batch_size):
        list_instance_batch.append(data_instances[i:i + args.train_batch_size])
    return list_instance_batch

def convert_examples_to_features1(args, train_instances, max_span_length=8):

    features = []
    num_aspect = 0
    num_quad = 0
    num_opinion = 0
    differ_opinion_senitment_num = 0
    categories,id2category = get_categories(args)
    category2id = {v: k for k, v in id2category.items()}
    for ex_index, example in enumerate(train_instances):
        sample = {'id': example.id}
        sample['tokens'] = example.text_a.split()
        sample['text_length'] = len(sample['tokens'])
        sample['quad'] = example.all_label
        if len(sample['quad'])==0:
            continue
        sample['sentence'] = example.text_a
        aspect = {}
        opinion = {}
        category = {}
        opinion_reverse = {}
        aspect_reverse  = {}
        spans_aspect2category_label = []
        sample['ao_pair'] = []
        sample['imp_asp'] = []
        sample['imp_opi'] = []
        sample['asp_sen'] = []
        sample['opi_sen'] = []
        for quad_name in sample['quad']:
            aspect_span, opinion_span, categorystr,sentiment = tuple(sample['quad'][quad_name][0]), tuple(
                sample['quad'][quad_name][1]), sample['quad'][quad_name][2],sample['quad'][quad_name][3]
            spans_aspect2category_label.append(category2id[categorystr])
            if aspect_span == (-1,-2):
                tem_a = [0,0,0]
                # sample['imp_asp'].append(1)
            else:
                tem_a = [aspect_span[0],aspect_span[1],aspect_span[1]+1-aspect_span[0]]
                # sample['imp_asp'].append(0)

            if opinion_span == (-1,-2):
                tem_o = [0,0,0]
                # sample['imp_opi'].append(1)
            else:
                tem_o = [opinion_span[0],opinion_span[1],opinion_span[1]+1-opinion_span[0]]
                # sample['imp_opi'].append(0)
            sample['ao_pair'].append([tem_a,tem_o])


            num_quad += 1
            if aspect_span not in aspect:
                aspect[aspect_span] = sentiment
                opinion[aspect_span] = [(opinion_span, sentiment)]
            else:
                opinion[aspect_span].append((opinion_span, sentiment))

            if opinion_span not in opinion_reverse:
                opinion_reverse[opinion_span] = sentiment
                # 1.category[opinion_span] = categorystr
                aspect_reverse[opinion_span] = [(aspect_span, sentiment)]
            else:
                # if opinion_reverse[opinion_span] != sentiment:
                #     differ_opinion_sentiment = True
                # else:
                #     if opinion_span == (-1,-2) and aspect_span == (-1,-2):
                #         print("存在双重隐性观点词和方面词："+sample['sentence'])
                aspect_reverse[opinion_span].append((aspect_span, sentiment))
                    # category[opinion_span].append(categorystr)
        # if differ_opinion_sentiment:
        #     differ_opinion_senitment_num += 1
        #     # print(ex_index, '单意见词多极性')
        #     continue
        temp_flag = 0
        aspect_polarity_label = []
        opinion_polarity_label = []
        temp_asp=[]
        temp_opi=[]
        if len(aspect_reverse)==0:
            temp_asp.append(0)
        for asp_span,opi_sen_list in aspect_reverse.items():
            for opi_sen in opi_sen_list:
                if opi_sen[0] == (-1,-2):
                    temp_asp.append(1)
                    aspect_polarity_label.append(sentiment2id[opi_sen[1]])
                    temp_flag = 1
                else:
                    temp_asp.append(0)
                    aspect_polarity_label.append(sentiment2id[opi_sen[1]])
            temp_flag = 0
        # assert len(opinion) == len(sample['imp_asp'])

        temp_flag = 0
        if len(opinion)==0:
            temp_opi.append(0)
        for opi_span, asp_sen_list in opinion.items():
            for asp_sen in asp_sen_list:
                if asp_sen[0] == (-1, -2):
                    temp_opi.append(1)
                    opinion_polarity_label.append(sentiment2id[asp_sen[1]])
                    temp_flag = 1
                else:
                    temp_opi.append(0)
                    opinion_polarity_label.append(sentiment2id[asp_sen[1]])
            temp_flag = 0
            #aspect_polarity_label.append(sentiment2id[asp_sentiment])

        if all(x in temp_asp for x in [0, 1]):
            sample['imp_asp'].append(1)
        elif 1 in temp_asp:
            sample['imp_asp'].append(1)
        else:
            sample['imp_asp'].append(0)

        if all(x in temp_opi for x in [0, 1]):
            sample['imp_opi'].append(1)
        elif 1 in temp_opi:
            sample['imp_opi'].append(1)
        else:
            sample['imp_opi'].append(0)
        '''
        for opi_span,opi_sentiment in opinion_reverse.items():
            if opi_span == (-1,-2):
                sample['imp_opi'].append(1)
            else:
                sample['imp_opi'].append(0)
            opinion_polarity_label.append(sentiment2id[opi_sentiment])
        num_aspect += len(aspect)
        num_opinion += len(opinion)
        '''
        # if len(aspect) != example.aspect_num:
        #     print('有不同三元组使用重复了aspect:', example.id)

        spans = []
        span_tokens = []

        spans_aspect_label = []
        spans_aspect2opinion_label =[]

        spans_opinion_label = []

        reverse_opinion_label = [0]
        reverse_opinion2aspect_label = []
        reverse_aspect_label = []


        # 解决隐式aspect及其对应的category label问题

        if (-1,-2) in aspect:
            spans_aspect_label.append(1)
            # aspect_polarity_label.append(sentiment2id[aspect[(-1,-2)]])
            # spans_aspect2category_label.append(category2id[category[(-1,-2)]])
        else:
            spans_aspect_label.append(0)
            # spans_aspect2category_label.append(0)

        if args.order_input:
            for i in range(max_span_length):
                if sample['text_length'] < i:
                    continue
                for j in range(sample['text_length'] - i):
                    # 所有可能的span（0,0,1）(1,1,1)....

                    # 当前span的token
                    span_token = ' '.join(sample['tokens'][j:i + j + 1])
                    if not is_valid_span(sample['tokens'][j:i + j + 1]):
                        continue
                    spans.append((j, i + j, i + 1))
                    # 所有的span token
                    span_tokens.append(span_token)
                    if (j, i + j) not in aspect:
                        # 当前span没有情感
                        spans_aspect_label.append(0)
                    else:
                        # aspect_polarity_label.append(sentiment2id[aspect[(j, i + j)]])
                        spans_aspect_label.append(validity2id[aspect[(j, i + j)]])
                    if (j, i + j) not in opinion_reverse:
                        # 当前opinion没有情感
                        reverse_opinion_label.append(0)
                    else:
                        # opinion_polarity_label.append(sentiment2id[opinion_reverse[(j, i + j)]])
                        # 对应到标签 添加情感
                        reverse_opinion_label.append(validity2id[opinion_reverse[(j, i + j)]])
                    # if (j , i + j) not in category:
                    #     spans_aspect2category_label.append(0)
                    # else:
                    #     spans_aspect2category_label.append(category2id[category[(j,i+j)]])

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
        # spans_aspect2category_label.append(0)
        spans_aspect_label.append(0)
        if (-1,-2) in opinion_reverse:
            reverse_opinion_label.append(1)
            # opinion_polarity_label.append(sentiment2id[opinion_reverse[(-1, -2)]])
        else:
            reverse_opinion_label.append(0)

        assert len(span_tokens) == len(spans)

        for key_aspect in opinion:
            opinion_list = []
            sentiment_opinion = []
            spans_aspect2opinion_label.append(key_aspect)
            for opinion_span_2_aspect in opinion[key_aspect]:
                opinion_list.append(opinion_span_2_aspect[0])
                sentiment_opinion.append(opinion_span_2_aspect[1])
            # assert len(set(sentiment_opinion)) == 1
            opinion_label2triple = [3]
            for i in spans:
                if (i[0], i[1]) not in opinion_list:
                    opinion_label2triple.append(3)
                else:
                    opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])

            if (-1, -2) in opinion_list:
                opinion_label2triple.append(sentiment2id[sentiment_opinion[0]])
            else:
                opinion_label2triple.append(3)
            spans_opinion_label.append(opinion_label2triple)

        for opinion_key in aspect_reverse:
            aspect_list = []
            sentiment_aspect = []
            reverse_opinion2aspect_label.append(opinion_key)
            for aspect_span_2_opinion in aspect_reverse[opinion_key]:
                aspect_list.append(aspect_span_2_opinion[0])
                sentiment_aspect.append(aspect_span_2_opinion[1])
            # assert len(set(sentiment_aspect)) == 1
            aspect_label2triple = []

            if (-1, -2) in aspect_list:
                aspect_label2triple.append(sentiment2id[sentiment_aspect[0]])
            else:
                aspect_label2triple.append(3)

            for i in spans:
                if (i[0], i[1]) not in aspect_list:
                    aspect_label2triple.append(3)
                else:
                    aspect_label2triple.append(sentiment2id[sentiment_aspect[0]])

            aspect_label2triple.append(3)

            reverse_aspect_label.append(aspect_label2triple)

        sample['aspect_num'] = len(spans_opinion_label)
        sample['spans_aspect2opinion_label'] = spans_aspect2opinion_label
        sample['reverse_opinion_num'] = len(reverse_aspect_label)
        sample['reverse_opinion2aspect_label'] = reverse_opinion2aspect_label
        sample['spans_aspect2category_label'] = spans_aspect2category_label
        sample['aspect_polarity_label'] = aspect_polarity_label
        sample['opinion_polarity_label'] = opinion_polarity_label
        related_spans = np.zeros((len(spans)+2, len(spans)+2), dtype=int)
        for i in range(len(span_tokens)):
            span_token = span_tokens[i].split(' ')
            # for j in range(i, len(span_tokens)):
            for j in range(len(span_tokens)):
                differ_span_token = span_tokens[j].split(' ')
                if set(span_token) & set(differ_span_token) == set():
                    related_spans[i+1, j+1] = 0
                else:
                    related_spans[i+1, j+1] = 1

        sample['related_span_array'] = related_spans
        sample['spans'], sample['span tokens'], sample['spans_aspect_label'], sample[
            'spans_opinion_label'] = spans, span_tokens, spans_aspect_label, spans_opinion_label
        sample['reverse_opinion_label'], sample['reverse_aspect_label'] = reverse_opinion_label, reverse_aspect_label
        features.append(sample)
    return features, num_aspect, num_opinion


def load_data_instances_txt1(lines,args):
    id2sentiment = {'0': 'negative', '1': 'neutral', '2': 'positive'}

    instances = list()
    quad_num = 0
    aspects_num = 0
    quad_num=0
    for ex_index, line in enumerate(lines):
        id = str(ex_index)  # id
        line = line.strip()
        line = line.split('\t')
        sentence = line[0].split()  # sentence
        # raw_pairs = eval(line[1])  # triplets
        raw_pairs = line[1:]
        # if len(raw_pairs)==0:
        #     continue;
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

            if  is_valid_span(aspect_word.split(' ')) and is_valid_span(opinion_word.split(' ')):
                # if aspect_word!=""  and opinion_word=="":
                word = str(aspect_word) + '|' + str(opinion_word) + '|'+str(raw_category)
                if word not in quad_dict:
                    quad_dict[word] = []
                    quad_dict[word] = ([raw_aspect[0], raw_aspect[-1]], [raw_opinion[0], raw_opinion[-1]], raw_category ,sentiment)
                    quad_num += 1

        examples = InputExample(id=id, text_a=line[0], text_b=None, all_label=quad_dict, aspect_num=aspect_num,
                                    triple_num=len(quad_dict))

        instances.append(examples)
        # quad_num += quad_num
        aspects_num += aspect_num

    print("quad_num:"+str(quad_num))
    return instances



class DataTterator2(object):
    def __init__(self, instances, args):
        # with open("./models/"+args.dataset + "_" + args.mode + "_output_new.json", 'r', encoding='utf-8') as f:
        #     raw_data = json.load(f)
        self.instances = instances
        self.args = args
        self.batch_count = len(instances)
        self.tokenizer = AutoTokenizer.from_pretrained(args.init_model,do_lower_case=args.do_lower_case)
        # self.adj = self.adj_compute(args, raw_data, self.tokenizer)


    

    def get_batch(self, batch_num):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_aspect_tensor_list = []
        spans_opinion_label_tensor_list = []

        spans_category_label_list = []

        reverse_ner_label_tensor_list = []
        reverse_opinion_tensor_list = []
        reverse_aspect_tensor_list = []
        sentence_length = []
        related_spans_list = []

        imp_asp_label_list = []
        imp_opi_label_list = []
        aspect_polarity_label_list = []
        opinion_polarity_label_list = []
        max_tokens = self.args.max_seq_length
        max_spans = 0
        for i, sample in enumerate(self.instances[batch_num]):
            tokens = sample['tokens']
            spans = sample['spans']
            span_tokens = sample['span tokens']
            spans_ner_label = sample['spans_aspect_label']#记录aspect位置
            spans_aspect2opinion_labels = sample['spans_aspect2opinion_label']#记录aspect对应的opinion元组
            spans_opinion_label = sample['spans_opinion_label']#记录opinion位置

            spans_category_label = sample['spans_aspect2category_label']

            reverse_ner_label = sample['reverse_opinion_label'] #记录opinion位置
            reverse_opinion2aspect_labels = sample['reverse_opinion2aspect_label']#记录opinion对应的aspect元组
            reverse_aspect_label = sample['reverse_aspect_label']# 记录aspect位置

            related_spans = sample['related_span_array']
            # spans_aspect_labels:[(batch_num,opinion_span1,opinion_span2)]
            ao_pair = sample['ao_pair']
            quad = sample['quad']

            imp_asp_label = sample['imp_asp']
            imp_opi_label = sample['imp_opi']

            aspect_polarity_label = sample['aspect_polarity_label']
            opinion_polarity_label = sample['opinion_polarity_label']

            spans_aspect_labels, reverse_opinion_labels = [], []
            for spans_aspect2opinion_label in spans_aspect2opinion_labels:
                spans_aspect_labels.append((i, spans_aspect2opinion_label[0], spans_aspect2opinion_label[1]))
            for reverse_opinion2aspect_label in reverse_opinion2aspect_labels:
                reverse_opinion_labels.append((i, reverse_opinion2aspect_label[0], reverse_opinion2aspect_label[1]))
            # bert_tokens, tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_labels_tensor, spans_opinion_tensor, \
            # reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor = \
            #     self.get_input_tensors(self.tokenizer, tokens, spans, spans_ner_label, spans_aspect_labels,
            #                              spans_opinion_label, reverse_ner_label, reverse_opinion_labels, reverse_aspect_label)
            # for j in range(len(spans_aspect2opinion_labels)):
            #     ao_pair.append((i,spans_aspect2opinion_labels[j],reverse_opinion2aspect_labels[j]))
            bert_tokens, tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_labels_tensor, spans_opinion_tensor, \
                reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor,spans_category_label_tensor,ao_pair, \
                imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, opinion_polarity_label_tensor= \
                self.get_input_tensors(self.tokenizer, tokens, spans, spans_ner_label, spans_aspect_labels,
                                       spans_opinion_label, reverse_ner_label, reverse_opinion_labels,
                                       reverse_aspect_label,spans_category_label,ao_pair,imp_asp_label,imp_opi_label,
                                       aspect_polarity_label,opinion_polarity_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            spans_aspect_tensor_list.append(spans_aspect_labels_tensor)
            spans_opinion_label_tensor_list.append(spans_opinion_tensor)
            reverse_ner_label_tensor_list.append(reverse_ner_label_tensor)
            reverse_opinion_tensor_list.append(reverse_opinion_tensor)
            reverse_aspect_tensor_list.append(reverse_aspect_tensor)
            spans_category_label_list.append(spans_category_label_tensor)

            imp_asp_label_list.append(imp_asp_label_tensor)
            imp_opi_label_list.append(imp_opi_label_tensor)
            aspect_polarity_label_list.append(aspect_polarity_label_tensor)
            opinion_polarity_label_list.append(opinion_polarity_label_tensor)
            # assert bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1] == reverse_ner_label_tensor.shape[1]
            # tokens和spans的最大个数被设定为固定值
            # if (tokens_tensor.shape[1] > max_tokens):
            #     max_tokens = tokens_tensor.shape[1]
            if (spans_ner_label_tensor.shape[1] > max_spans):
                max_spans = spans_ner_label_tensor.shape[1]
            sentence_length.append((bert_tokens, tokens_tensor.shape[1], spans_ner_label_tensor.shape[1],ao_pair,spans_category_label_tensor,quad))
            related_spans_list.append(related_spans)
        '''由于不同句子方阵不一样大，所以先不转为tensor'''
        # related_spans_tensor = torch.tensor(related_spans_list)
        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_spans_mask_tensor = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_aspect_tensor = None
        final_spans_opinion_label_tensor = None

       # final_spans_category_label_tensor = None

        final_reverse_ner_label_tensor = None
        final_reverse_opinion_tensor = None
        final_reverse_aspect_label_tensor = None
        final_related_spans_tensor = None

        final_imp_asp_label_tensor = None
        final_imp_opi_label_tensor = None
        final_aspect_polarity_label_tensor = None
        final_opinion_polarity_label_tensor = None


        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_tensor, spans_opinion_label_tensor, \
            reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor, related_spans,spans_category_label_tensor, \
            imp_asp_label_tensor, imp_opi_label_tensor, aspect_polarity_label_tensor, opinion_polarity_label_tensor \
                in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list, spans_aspect_tensor_list,
                       spans_opinion_label_tensor_list, reverse_ner_label_tensor_list, reverse_opinion_tensor_list,
                       reverse_aspect_tensor_list, related_spans_list,spans_category_label_list,
                       imp_asp_label_list,imp_opi_label_list,aspect_polarity_label_list,opinion_polarity_label_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = spans_ner_label_tensor.shape[1]
            num_aspect = spans_aspect_tensor.shape[1]
            num_opinion = reverse_opinion_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)

                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                reverse_ner_label_tensor = torch.cat((reverse_ner_label_tensor, mask_pad), dim=1)

                spans_category_label_tensor = torch.cat((spans_category_label_tensor,mask_pad),dim=1)
                if num_aspect !=0:
                    opinion_mask_pad = torch.full([1, num_aspect, spans_pad_length], 3, dtype=torch.long)
                    spans_opinion_label_tensor = torch.cat((spans_opinion_label_tensor, opinion_mask_pad), dim=-1)
                    aspect_mask_pad = torch.full([1, num_opinion, spans_pad_length], 3, dtype=torch.long)
                    reverse_aspect_tensor = torch.cat((reverse_aspect_tensor, aspect_mask_pad), dim=-1)
                '''对span类似方阵mask'''
                related_spans_pad_1 = np.zeros([num_spans, spans_pad_length])
                related_spans_pad_2 = np.zeros([spans_pad_length, max_spans])
                related_spans_hstack = np.hstack((related_spans, related_spans_pad_1))
                related_spans = np.vstack((related_spans_hstack, related_spans_pad_2))
            related_spans_tensor = torch.as_tensor(torch.from_numpy(related_spans), dtype=torch.bool)
            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor

                final_bert_spans_tensor = bert_spans_tensor
                final_spans_mask_tensor = spans_mask_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor

                final_imp_asp_label_tensor = imp_asp_label_tensor.squeeze(0)
                final_imp_opi_label_tensor = imp_opi_label_tensor.squeeze(0)
               # final_spans_category_label_tensor = spans_category_label_tensor
                final_related_spans_tensor = related_spans_tensor.unsqueeze(0)

                final_spans_aspect_tensor = spans_aspect_tensor.squeeze(0)
                final_spans_opinion_label_tensor = spans_opinion_label_tensor.squeeze(0)
                final_reverse_ner_label_tensor = reverse_ner_label_tensor
                final_reverse_opinion_tensor = reverse_opinion_tensor.squeeze(0)
                final_reverse_aspect_label_tensor = reverse_aspect_tensor.squeeze(0)
                final_aspect_polarity_label_tensor = aspect_polarity_label_tensor.squeeze(0)
                final_opinion_polarity_label_tensor = opinion_polarity_label_tensor.squeeze(0)

            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                #final_spans_category_label_tensor = torch.cat((final_spans_category_label_tensor, spans_category_label_tensor), dim=0)
                final_reverse_ner_label_tensor = torch.cat(
                    (final_reverse_ner_label_tensor, reverse_ner_label_tensor), dim=0)
                final_related_spans_tensor = torch.cat(
                    (final_related_spans_tensor, related_spans_tensor.unsqueeze(0)), dim=0)
                final_imp_asp_label_tensor = torch.cat((final_imp_asp_label_tensor, imp_asp_label_tensor.squeeze(0)),
                                                       dim=0)
                final_imp_opi_label_tensor = torch.cat((final_imp_opi_label_tensor, imp_opi_label_tensor.squeeze(0)),
                                                       dim=0)


                final_spans_aspect_tensor = torch.cat(
                    (final_spans_aspect_tensor, spans_aspect_tensor.squeeze(0)), dim=0).to(torch.long)
                final_spans_opinion_label_tensor = torch.cat(
                    (final_spans_opinion_label_tensor, spans_opinion_label_tensor.squeeze(0)), dim=0).to(torch.long)
                final_reverse_opinion_tensor = torch.cat(
                    (final_reverse_opinion_tensor,  reverse_opinion_tensor.squeeze(0)), dim=0).to(torch.long)
                final_reverse_aspect_label_tensor = torch.cat(
                    (final_reverse_aspect_label_tensor, reverse_aspect_tensor.squeeze(0)), dim=0).to(torch.long)
                final_aspect_polarity_label_tensor = torch.cat((final_aspect_polarity_label_tensor,aspect_polarity_label_tensor.squeeze(0)),dim=0).to(torch.long)
                final_opinion_polarity_label_tensor = torch.cat((final_opinion_polarity_label_tensor,opinion_polarity_label_tensor.squeeze(0)),dim=0).to(torch.long)

        # 注意，特征中最大span间隔不一定为设置的max_span_length，这是因为bert分词之后造成的span扩大了。
        final_tokens_tensor = final_tokens_tensor.to(self.args.device)
        final_attention_mask = final_attention_mask.to(self.args.device)
        final_bert_spans_tensor = final_bert_spans_tensor.to(self.args.device)
        final_spans_mask_tensor = final_spans_mask_tensor.to(self.args.device)
        final_spans_ner_label_tensor = final_spans_ner_label_tensor.to(self.args.device)

       # final_spans_category_label_tensor = final_spans_category_label_tensor.to(self.args.device)

        final_spans_aspect_tensor = final_spans_aspect_tensor.to(self.args.device)
        final_spans_opinion_label_tensor = final_spans_opinion_label_tensor.to(self.args.device)
        final_reverse_ner_label_tensor = final_reverse_ner_label_tensor.to(self.args.device)
        final_reverse_opinion_tensor = final_reverse_opinion_tensor.to(self.args.device)
        final_reverse_aspect_label_tensor = final_reverse_aspect_label_tensor.to(self.args.device)
        final_related_spans_tensor = final_related_spans_tensor.to(self.args.device)

        final_imp_asp_label_tensor = final_imp_asp_label_tensor.to(self.args.device)
        final_imp_opi_label_tensor = final_imp_opi_label_tensor.to(self.args.device)
        final_aspect_polarity_label_tensor = final_aspect_polarity_label_tensor.to(self.args.device)
        final_opinion_polarity_label_tensor = final_opinion_polarity_label_tensor.to(self.args.device)


        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, \
               final_spans_ner_label_tensor, final_spans_aspect_tensor, final_spans_opinion_label_tensor, \
               final_reverse_ner_label_tensor, final_reverse_opinion_tensor, final_reverse_aspect_label_tensor, \
               final_related_spans_tensor, sentence_length,final_imp_asp_label_tensor,final_imp_opi_label_tensor, \
               final_aspect_polarity_label_tensor,final_opinion_polarity_label_tensor


    def get_input_tensors(self, tokenizer, tokens, spans, spans_ner_label, spans_aspect_labels, spans_opinion_label,
                          reverse_ner_label, reverse_opinion_labels, reverse_aspect_label,spans_category_label,ao_pair,
                          imp_asp_label, imp_opi_label,aspect_polarity_label, opinion_polarity_label):
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens.append("[CLS]")
        for token in tokens:
            if token == '':
                continue
            start2idx.append(len(bert_tokens))
            test_1 = len(bert_tokens)
            sub_tokens = tokenizer.tokenize(token)
            if self.args.span_generation == "CNN":
                bert_tokens.append(sub_tokens[0])
            elif self.args.Only_token_head:
                bert_tokens.append(sub_tokens[0])
            else:
                bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
            test_2 = len(bert_tokens) - 1
            # if test_2 != test_1:
            #     print("差异：", test_2 - test_1)
            # else:
            #     print("no extra token")
        # start2idx.append(len(bert_tokens))
        # end2idx.append(len(bert_tokens))
        bert_tokens.append("[CLS]")
        # bert_tokens.append(tokenizer.sep_token)
        indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        final_ao_pair = []
        for span in ao_pair:
            tem_a = [start2idx[span[0][0]], end2idx[span[0][1]], span[0][2]] if span[0] != [0, 0, 0] else [0, 0, 0]
            tem_o = [start2idx[span[1][0]], end2idx[span[1][1]], span[1][2]] if span[1] != [0, 0, 0] else [0, 0, 0]
            final_ao_pair.append([tem_a, tem_o])

        bert_spans.append([0,0,0])
        bert_spans.insert(0,[0,0,0])
        # 在bert分出subword之后  需要对原有的aspect span进行补充
        spans_aspect_label = []
        reverse_opinion_label = []
        for i,aspect_span in enumerate(spans_aspect_labels):
            if aspect_span[1] == -1 and aspect_span[2] == -2:
                spans_aspect_label.append([aspect_span[0], -1, -1])
            else:
                spans_aspect_label.append([aspect_span[0],
                                           start2idx[aspect_span[1]],
                                           end2idx[aspect_span[2]]])
                if start2idx[aspect_span[1]]>end2idx[aspect_span[2]]:
                    print("error aspect")

        for i,opinion_span in enumerate(reverse_opinion_labels):
            if opinion_span[1] == -1 and opinion_span[2] == -2:
                reverse_opinion_label.append([opinion_span[0], -1, -1])
            else:
                reverse_opinion_label.append([opinion_span[0],
                                              start2idx[opinion_span[1]],
                                              end2idx[opinion_span[2]]])
                if  start2idx[opinion_span[1]]>end2idx[opinion_span[2]]:
                    print("error opinion")
        # spans_aspect_label = [[aspect_span[0], start2idx[aspect_span[1]], end2idx[aspect_span[2]]] for
        #                       aspect_span in spans_aspect_label]
        # reverse_opinion_label =[[opinion_span[0], start2idx[opinion_span[1]], end2idx[opinion_span[2]]] for
        #                         opinion_span in reverse_opinion_labels]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        spans_aspect_tensor = torch.tensor([spans_aspect_label])
        spans_opinion_tensor = torch.tensor([spans_opinion_label])
        reverse_ner_label_tensor = torch.tensor([reverse_ner_label])
        reverse_opinion_tensor = torch.tensor([reverse_opinion_label])
        reverse_aspect_tensor = torch.tensor([reverse_aspect_label])
        spans_category_label_tensor = torch.tensor([spans_category_label])

        imp_asp_label_tensor = torch.tensor([imp_asp_label])
        imp_opi_label_tensor = torch.tensor([imp_opi_label])
        aspect_polarity_label_tensor = torch.tensor([aspect_polarity_label])
        opinion_polarity_label_tensor = torch.tensor([opinion_polarity_label])

        return bert_tokens,tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_aspect_tensor, spans_opinion_tensor, \
               reverse_ner_label_tensor, reverse_opinion_tensor, reverse_aspect_tensor,spans_category_label_tensor,final_ao_pair, \
               imp_asp_label_tensor,imp_opi_label_tensor,aspect_polarity_label_tensor,opinion_polarity_label_tensor



if __name__ == '__main__':
    write_json_data("../datasets/Laptop-ACOS/laptop_quad_train.tsv","laptop_trian")
    write_json_data("../datasets/Laptop-ACOS/laptop_quad_test.tsv","laptop_test")
    write_json_data("../datasets/Laptop-ACOS/laptop_quad_dev.tsv","laptop_dev")

    write_json_data("../datasets/Restaurant-ACOS/rest16_quad_train.tsv","restaurant_train")
    write_json_data("../datasets/Restaurant-ACOS/rest16_quad_test.tsv", "restaurant_test")
    write_json_data("../datasets/Restaurant-ACOS/rest16_quad_dev.tsv", "restaurant_dev")
