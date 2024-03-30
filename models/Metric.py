from transformers import BertTokenizer
import numpy as np
import json
from .data_BIO_loader import get_categories


id4validity = {0: 'none', 1: 'valid'}
id4sentiment = {3: 'none', 2: 'positive', 0: 'negative', 1:'neutral'}
# categories = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES',
#             'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']

class Metric():
    def __init__(self, args, forward_pred_result, reverse_pred_result, gold_instances):
        self.args = args
        self.gold_instances = gold_instances
        self.tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)

        self.pred_aspect = forward_pred_result[0]
        self.pred_aspect_sentiment = forward_pred_result[1]
        self.pred_aspect_sentiment_logit = forward_pred_result[2]

        self.pred_opinion = forward_pred_result[3]
        self.pred_opinion_sentiment_logit = forward_pred_result[4]

        self.pred_aspect_category = forward_pred_result[6]
        '''反向评价'''
        self.reverse_pred_opinon = reverse_pred_result[0]
        self.reverse_pred_opinon_sentiment = reverse_pred_result[1]
        self.reverse_pred_opinon_sentiment_logit = reverse_pred_result[2]

        self.reverse_pred_aspect = reverse_pred_result[3]
        self.reverse_pred_aspect_sentiment_logit = reverse_pred_result[4]

        self.imp_opi_asp = reverse_pred_result[5]
        self.imp_asp_opi = forward_pred_result[5]
    def P_R_F1(self, gold_num, pred_num, correct_num):
        precision = correct_num / pred_num if pred_num > 0 else 0
        recall = correct_num / gold_num if gold_num > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return (precision, recall, f1)

    def num_4_eval(self, gold, pred, gold_num, pred_num, correct_num):
        correct = set(gold) & set(pred)
        gold_num += len(set(gold))
        pred_num += len(set(pred))
        correct_num += len(correct)
        return gold_num, pred_num, correct_num

    def cal_triplet_final_result(self, forward_results, forward_spans, reverse_results, reverse_spans):

        pred_dicts = {}
        pred_spans = forward_spans + reverse_spans
        for index, result in enumerate(forward_results + reverse_results):
            if result in pred_dicts:
                score_dict = pred_dicts[result][2]
                score_new = pred_spans[index][2]
                if score_dict > score_new:
                    continue
                else:
                    pred_dicts[result] = pred_spans[index]
            else:
                pred_dicts[result] = pred_spans[index]
        history = []
        for i in pred_dicts:
            aspect_span_i = range(pred_dicts[i][0][0], pred_dicts[i][0][1])
            opinion_span_i = range(pred_dicts[i][1][0], pred_dicts[i][1][1])
            for j in pred_dicts:
                if (i,j) in history:
                    continue
                history.append((i, j))
                history.append((j, i))
                if i == j:
                    continue
                aspect_span_j = range(pred_dicts[j][0][0], pred_dicts[j][0][1])
                opinion_span_j = range(pred_dicts[j][1][0], pred_dicts[j][1][1])
                repeat_a_span = list(set(aspect_span_i) & set(aspect_span_j))
                repeat_o_span = list(set(opinion_span_i) & set(opinion_span_j))
                if len(repeat_a_span) == 0 or len(repeat_o_span) == 0:
                    continue
                elif len(repeat_a_span) <= min(len(aspect_span_i), len(aspect_span_j)) and \
                        len(repeat_o_span) <= min(len(opinion_span_i), len(opinion_span_j)):
                    i_score = pred_dicts[i][2]
                    j_score = pred_dicts[j][2]
                    if i_score >= j_score:
                        pred_dicts[j] = (pred_dicts[j][0], pred_dicts[j][1], 0)
                    else:
                        pred_dicts[i] = (pred_dicts[i][0], pred_dicts[i][1], 0)
                else:
                    raise(KeyboardInterrupt)
        return [_ for _ in pred_dicts if pred_dicts[_][2] != 0]

    def cal_quad_final_result(self, forward_results, forward_spans, reverse_results, reverse_spans):

        pred_dicts = {}
        # final_quad = list(forward_results)
        # for element in reverse_results:
        #     first_element = element[0]
        #     for element_f in forward_results:
        #         if first_element == element_f[0]:
        #             final_quad.append((element[0],element[1],element_f[2],element[2]))

        pred_spans = forward_spans + reverse_spans
        for index, result in enumerate(forward_results + reverse_results):
            if result in pred_dicts:
                score_dict = pred_dicts[result][2]
                score_new = pred_spans[index][2]
                if score_dict > score_new:
                    continue
                else:
                    pred_dicts[result] = pred_spans[index]
            else:
                pred_dicts[result] = pred_spans[index]
        history = []
        for i in pred_dicts:
            aspect_span_i = range(pred_dicts[i][0][0], pred_dicts[i][0][1])
            opinion_span_i = range(pred_dicts[i][1][0], pred_dicts[i][1][1])
            for j in pred_dicts:
                if (i, j) in history:
                    continue
                history.append((i, j))
                history.append((j, i))
                if i == j:
                    continue
                aspect_span_j = range(pred_dicts[j][0][0], pred_dicts[j][0][1])
                opinion_span_j = range(pred_dicts[j][1][0], pred_dicts[j][1][1])
                repeat_a_span = list(set(aspect_span_i) & set(aspect_span_j))
                repeat_o_span = list(set(opinion_span_i) & set(opinion_span_j))
                if len(repeat_a_span) == 0 or len(repeat_o_span) == 0:
                    continue
                elif len(repeat_a_span) <= min(len(aspect_span_i), len(aspect_span_j)) and \
                        len(repeat_o_span) <= min(len(opinion_span_i), len(opinion_span_j)):
                    i_score = pred_dicts[i][2]
                    j_score = pred_dicts[j][2]
                    if i_score >= j_score:
                        pred_dicts[j] = (pred_dicts[j][0], pred_dicts[j][1], 0)
                    else:
                        pred_dicts[i] = (pred_dicts[i][0], pred_dicts[i][1], 0)
                else:
                    raise (KeyboardInterrupt)
        return [_ for _ in pred_dicts if pred_dicts[_][2] != 0]

    def merge_lists(self,list1, list2):
        merged_list = []

        # 将两个列表合并
        combined_list = list1 + list2

        # 使用字典来存储每个元组的前两位对应的最大第三位值
        merged_dict = {}

        for item in combined_list:
            key = (item[0], item[1])
            if key in merged_dict:
                # 如果已经存在相同的键，更新对应的值为最大值
                merged_dict[key] = max(merged_dict[key], item[2])
            else:
                # 如果键不存在，直接添加到字典
                merged_dict[key] = item[2]

        # 将字典中的键值对还原为元组，并添加到结果列表中
        for key, value in merged_dict.items():
            merged_list.append((key[0], key[1], value))

        return merged_list

    def num_pair_eval(self,gold_pairs_num, pred_pairs_num, correct_pairs_num, low_standard,pair, gold_pairs):
        gold_pairs_num += len(set(gold_pairs))
        pred_pairs_num += len(set(pair))

        for temp in pair:
            # if temp[2]<0.88:
            #     # print(pairs, temp[2])
            pairs = (temp[0],temp[1])

            if pairs in gold_pairs:
                correct_pairs_num += 1
                if temp[2]<low_standard:
                    low_standard = temp[2]

        return gold_pairs_num, pred_pairs_num, correct_pairs_num, low_standard
    def score_triples(self):
        correct_aspect_num,correct_opinion_num,correct_apce_num,correct_pairs_num,correct_num = 0,0,0,0,0
        gold_aspect_num,gold_opinion_num,gold_apce_num,gold_pairs_num,gold_num = 0,0,0,0,0
        pred_aspect_num,pred_opinion_num,pred_apce_num,pred_pairs_num,pred_num = 0,0,0,0,0
        cat_num = 0
        low_standard = 0.95
        # gold_aspect_num_length1, gold_aspect_num_length2, gold_aspect_num_length3, gold_aspect_num_length4, gold_aspect_num_length5 = 0,0,0,0,0
        # pred_aspect_num_length1, pred_aspect_num_length2, pred_aspect_num_length3, pred_aspect_num_length4, pred_aspect_num_length5 = 0,0,0,0,0
        # correct_aspect_num_length1, correct_aspect_num_length2, correct_aspect_num_length3, correct_aspect_num_length4, correct_aspect_num_length5 =  0,0,0,0,0
        #
        # gold_opinion_num_length1, gold_opinion_num_length2, gold_opinion_num_length3, gold_opinion_num_length4, gold_opinion_num_length5 = 0,0,0,0,0
        # pred_opinion_num_length1, pred_opinion_num_length2, pred_opinion_num_length3, pred_opinion_num_length4, pred_opinion_num_length5 = 0,0,0,0,0
        # correct_opinion_num_length1, correct_opinion_num_length2, correct_opinion_num_length3, correct_opinion_num_length4, correct_opinion_num_length5 =  0,0,0,0,0
        categories,id2category = get_categories(self.args)
        if self.args.output_path:
            result = []
            result1= []
            result2 = []
            all_result = []
            triples_result = []
            aspect_text = []
            opinion_text = []
        for i in range(len(self.gold_instances)):
            '''实体长度实验'''
            # gold_aspect_length1, gold_aspect_length2, gold_aspect_length3, gold_aspect_length4, gold_aspect_length5 = [], [], [], [], []
            # pred_aspect_length1, pred_aspect_length2, pred_aspect_length3, pred_aspect_length4, pred_aspect_length5 = [], [], [], [], []
            #
            # gold_opinion_length1, gold_opinion_length2, gold_opinion_length3, gold_opinion_length4, gold_opinion_length5 = [], [], [], [], []
            # pred_opinion_length1, pred_opinion_length2, pred_opinion_length3, pred_opinion_length4, pred_opinion_length5 = [], [], [], [], []


            bert_tokens = []
            spans = self.gold_instances[i]['spans']
            start2idx = []
            end2idx = []
            bert_tokens.append(self.tokenizer.cls_token)
            for token in self.gold_instances[i]['tokens']:
                start2idx.append(len(bert_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                if self.args.span_generation == "CNN":
                    bert_tokens.append(sub_tokens[0])
                elif self.args.Only_token_head:
                    bert_tokens.append(sub_tokens[0])
                else:
                    bert_tokens += sub_tokens
                end2idx.append(len(bert_tokens) - 1)
            bert_tokens.append(self.tokenizer.cls_token)
            bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
            bert_spans.append([0, 0, 0])
            bert_spans.insert(0, [0, 0, 0])
            gold_aspect, gold_opinion, gold_apce, gold_pairs, gold_triples,gold_quad,gold_category = self.find_gold_quad(i, bert_spans,
                                                                                                    bert_tokens)
            pred_aspect, pred_opinion, pred_apce, pred_pairs, pred_triples, pred_spans,pred_quad,pred_category = self.find_pred_quad(i, bert_spans,
                                                                                                    bert_tokens)


            reverse_aspect, reverse_opinion, reverse_apce, reverse_pairs, reverse_triples, reverse_spans = \
                self.find_pred_reverse_triples(i, bert_spans, bert_tokens)

            # pred_aspect = list(set(pred_aspect) | set(reverse_aspect))
            # pred_opinion = list(set(pred_opinion) | set(reverse_opinion))
            # pred_apce = list(set(pred_apce) | set(reverse_apce))
            # pred_pairs = list(set(pred_pairs) | set(reverse_pairs))
            # pred_category = list(set(pred_category))

            if self.args.Filter_Strategy:
                pred_triples = self.cal_quad_final_result(pred_triples, pred_spans, reverse_triples, reverse_spans)
            else:
                pred_triples = list(set(pred_triples) | set(reverse_triples))



            pred_category_span = self.pred_aspect_category[i]
            quad_list = []
            if len(pred_category_span) > 0:
                #pred_category_span = list({tuple(data): data for data in pred_category_span}.values()) #去重
                for j, a_o_c in enumerate(pred_category_span):
                    aspect_span = bert_spans[a_o_c[0]]
                    aspect_span = (aspect_span[2], aspect_span[0], aspect_span[1])

                    opinion_span = bert_spans[int(a_o_c[1])]
                    opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                    aspect = self.find_token(bert_tokens, aspect_span)
                    opinion = self.find_token(bert_tokens, opinion_span)
                    for j, triple in enumerate(pred_triples):
                        if triple[0] == aspect and triple[1] == opinion:
                            quad_list.append((aspect, opinion, id2category[a_o_c[2]], triple[2]))
                            break
            '''
            for j, triple in enumerate(pred_triples):
                m = 0
                for j, a_o_c in enumerate(pred_category_span):
                    aspect_span = bert_spans[a_o_c[0]]
                    aspect_span = (aspect_span[2], aspect_span[0], aspect_span[1])

                    opinion_span = bert_spans[int(a_o_c[1])]
                    opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                    aspect = self.find_token(bert_tokens, aspect_span)
                    opinion = self.find_token(bert_tokens, opinion_span)
                    if triple[0] == aspect and triple[1] == opinion:
                        m=1
                        break
                if m == 0:
                    print(triple[0],triple[1])
            # quad_list = list(set(quad_list) | set(pred_quad))
            '''
            flag = 0
            for imp_quad in pred_quad:
                for quad in quad_list:
                    if quad[0]==imp_quad[0] and quad[1] == imp_quad[1]:
                        flag = 1
                if flag == 0:
                    quad_list.append(imp_quad)
                flag = 0

            pair = self.merge_lists(pred_pairs, reverse_pairs)
            gold_pairs_num, pred_pairs_num, correct_pairs_num, low_standard = self.num_pair_eval(gold_pairs_num,
                                                                                                 pred_pairs_num,
                                                                                                 correct_pairs_num,
                                                                                                 low_standard, pair,
                                                                                                 gold_pairs)
            gold_num, pred_num, correct_num = self.num_4_eval(gold_quad, quad_list, gold_num,
                                                              pred_num, correct_num)
            gold_apce_num, pred_apce_num, correct_apce_num = self.num_4_eval(gold_triples, pred_triples, gold_apce_num,
                                                                        pred_apce_num, correct_apce_num)
            if self.args.output_path:
                if len(gold_quad)<2:
                    result1.append({"sentence": self.gold_instances[i]['sentence'],
                                   "quad_list_gold": [gold_quad for gold_quad in set(gold_quad)],
                                   "quad_list_pred": [pred_quad for pred_quad in set(quad_list)],
                                   "new": [new_quad for new_quad in (set(quad_list) - set(gold_quad))],
                                   "lack": [lack_quad for lack_quad in (set(gold_quad) - set(quad_list))]
                                   })
                elif len(gold_quad) < 3:
                    result.append({"sentence": self.gold_instances[i]['sentence'],
                                         "quad_list_gold": [gold_quad for gold_quad in set(gold_quad)],
                                         "quad_list_pred": [pred_quad for pred_quad in set(quad_list)],
                                        "new": [new_quad for new_quad in (set(quad_list) - set(gold_quad))],
                                        "lack": [lack_quad for lack_quad in (set(gold_quad) - set(quad_list))]
                                         })
                else:
                    result2.append({"sentence": self.gold_instances[i]['sentence'],
                                   "quad_list_gold": [gold_quad for gold_quad in set(gold_quad)],
                                   "quad_list_pred": [pred_quad for pred_quad in set(quad_list)],
                                   "new": [new_quad for new_quad in (set(quad_list) - set(gold_quad))],
                                   "lack": [lack_quad for lack_quad in (set(gold_quad) - set(quad_list))]
                                   })
                all_result.append({"sentence": self.gold_instances[i]['sentence'],
                                   "quad_list_gold": [gold_quad for gold_quad in set(gold_quad)],
                                   "quad_list_pred": [pred_quad for pred_quad in set(quad_list)],
                                   "new": [new_quad for new_quad in (set(quad_list) - set(gold_quad))],
                                   "lack": [lack_quad for lack_quad in (set(gold_quad) - set(quad_list))]
                                   })
            if self.args.output_path:
                triples_result.append({"sentence": self.gold_instances[i]['sentence'],
                                   "quad_list_gold": [gold_triples for gold_triples in set(gold_triples)],
                                   "quad_list_pred": [pred_triples for pred_triples in set(pred_triples)],
                                   "new": [new_triple for new_triple in (set(pred_triples) - set(gold_triples))],
                                   "lack": [lack_triple for lack_triple in (set(gold_triples) - set(pred_triples))]
                                   })
            # # pred_aspect = reverse_aspect
            # # pred_opinion = reverse_opinion
            # # pred_apce = reverse_apce
            # # pred_pairs = reverse_pairs
            # # pred_triples = reverse_triples


        if self.args.output_path:
            F = open(self.args.dataset + 'quad_num_2.json', 'w', encoding='utf-8')
            json.dump(result, F, ensure_ascii=False, indent=4)
            F.close()

            F1 = open(self.args.dataset + 'quad_num_3-5.json', 'w', encoding='utf-8')
            json.dump(result2, F1, ensure_ascii=False, indent=4)
            F1.close()
            F2 = open(self.args.dataset + 'quad_num_1.json', 'w', encoding='utf-8')
            json.dump(result1, F2, ensure_ascii=False, indent=4)
            F2.close()

            F3 = open(self.args.dataset + 'quad_num_all.json', 'w', encoding='utf-8')
            json.dump(all_result, F3, ensure_ascii=False, indent=4)
            F3.close()

        if self.args.output_path:
            # F = open(self.args.dataset + 'quad_num_2.json', 'w', encoding='utf-8')
            # json.dump(result, F, ensure_ascii=False, indent=4)
            # F.close()
            #
            # F1 = open(self.args.dataset + 'quad_num_3-5.json', 'w', encoding='utf-8')
            # json.dump(result2, F1, ensure_ascii=False, indent=4)
            # F1.close()
            # F2 = open(self.args.dataset + 'quad_num_1.json', 'w', encoding='utf-8')
            # json.dump(result1, F2, ensure_ascii=False, indent=4)
            # F2.close()

            F4 = open(self.args.dataset + 'triples_num_all.json', 'w', encoding='utf-8')
            json.dump(triples_result, F4, ensure_ascii=False, indent=4)
            F4.close()
        quad_result = self.P_R_F1(gold_num, pred_num, correct_num)
        pair_result = self.P_R_F1(gold_pairs_num, pred_pairs_num, correct_pairs_num,)
        triples_result = self.P_R_F1(gold_apce_num, pred_apce_num, correct_apce_num)
            # gold_aspect_num, pred_aspect_num, correct_aspect_num = self.num_4_eval(gold_aspect, pred_aspect,
            #                                                                        gold_aspect_num,
            #                                                                        pred_aspect_num, correct_aspect_num)
            #
            # gold_opinion_num, pred_opinion_num, correct_opinion_num = self.num_4_eval(gold_opinion, pred_opinion,
            #                                                                        gold_opinion_num,
            #                                                                        pred_opinion_num, correct_opinion_num)
            #
            # gold_apce_num, pred_apce_num, correct_apce_num = self.num_4_eval(gold_apce, pred_apce, gold_apce_num,
            #                                                                  pred_apce_num, correct_apce_num)
            #
            # gold_apce_num, pred_apce_num, correct_apce_num = self.num_4_eval(gold_apce, pred_apce, gold_apce_num,
            #                                                                  pred_apce_num, correct_apce_num)
            #
            # gold_pairs_num, pred_pairs_num, correct_pairs_num = self.num_4_eval(gold_pairs, pred_pairs, gold_pairs_num,
            #                                                                  pred_pairs_num, correct_pairs_num)

            # gold_num, pred_num, correct_num = self.num_4_eval(gold_triples, pred_triples, gold_num,
            #                                                                     pred_num, correct_num)


            # for aspect_G in gold_aspect:
            #     aspect_length = aspect_G.split(' ')
            #     if len(aspect_length) == 1:
            #         gold_aspect_length1.append(aspect_G)
            #     elif len(aspect_length) == 2:
            #         gold_aspect_length2.append(aspect_G)
            #     elif len(aspect_length) >= 3:
            #         gold_aspect_length3.append(aspect_G)
            #     elif len(aspect_length) == 4:
            #         gold_aspect_length4.append(aspect_G)
            #     elif len(aspect_length) >= 5:
            #         gold_aspect_length5.append(aspect_G)
            #
            # for aspect_P in pred_aspect:
            #     aspect_length = aspect_P.split(' ')
            #     if len(aspect_length) == 1:
            #         pred_aspect_length1.append(aspect_P)
            #     elif len(aspect_length) == 2:
            #         pred_aspect_length2.append(aspect_P)
            #     elif len(aspect_length) >= 3:
            #         pred_aspect_length3.append(aspect_P)
            #     elif len(aspect_length) == 4:
            #         pred_aspect_length4.append(aspect_P)
            #     elif len(aspect_length) >= 5:
            #         pred_aspect_length5.append(aspect_P)
            #
            # gold_aspect_num_length1, pred_aspect_num_length1, correct_aspect_num_length1 = self.num_4_eval(
            #     gold_aspect_length1, pred_aspect_length1, gold_aspect_num_length1, pred_aspect_num_length1, correct_aspect_num_length1)
            # gold_aspect_num_length2, pred_aspect_num_length2, correct_aspect_num_length2 = self.num_4_eval(
            #     gold_aspect_length2, pred_aspect_length2, gold_aspect_num_length2, pred_aspect_num_length2, correct_aspect_num_length2)
            # gold_aspect_num_length3, pred_aspect_num_length3, correct_aspect_num_length3 = self.num_4_eval(
            #     gold_aspect_length3, pred_aspect_length3, gold_aspect_num_length3, pred_aspect_num_length3, correct_aspect_num_length3)
            # gold_aspect_num_length4, pred_aspect_num_length4, correct_aspect_num_length4 = self.num_4_eval(
            #     gold_aspect_length4, pred_aspect_length4, gold_aspect_num_length4, pred_aspect_num_length4, correct_aspect_num_length4)
            # gold_aspect_num_length5, pred_aspect_num_length5, correct_aspect_num_length5 = self.num_4_eval(
            #     gold_aspect_length5, pred_aspect_length5, gold_aspect_num_length5, pred_aspect_num_length5, correct_aspect_num_length5)
            # assert gold_aspect_num_length1+gold_aspect_num_length2+gold_aspect_num_length3+gold_aspect_num_length4+gold_aspect_num_length5 == gold_aspect_num
            #
            # for opinion_G in gold_opinion:
            #     opinion_length = opinion_G.split(' ')
            #     if len(opinion_length) == 1:
            #         gold_opinion_length1.append(opinion_G)
            #     elif len(opinion_length) == 2:
            #         gold_opinion_length2.append(opinion_G)
            #     elif len(opinion_length) >= 3:
            #         gold_opinion_length3.append(opinion_G)
            #     elif len(opinion_length) == 4:
            #         gold_opinion_length4.append(opinion_G)
            #     elif len(opinion_length) >= 5:
            #         gold_opinion_length5.append(opinion_G)
            # for opinion_P in pred_opinion:
            #     opinion_length = opinion_P.split(' ')
            #     if len(opinion_length) == 1:
            #         pred_opinion_length1.append(opinion_P)
            #     elif len(opinion_length) == 2:
            #         pred_opinion_length2.append(opinion_P)
            #     elif len(opinion_length) >= 3:
            #         pred_opinion_length3.append(opinion_P)
            #     elif len(opinion_length) == 4:
            #         pred_opinion_length4.append(opinion_P)
            #     elif len(opinion_length) >= 5:
            #         pred_opinion_length5.append(opinion_P)
            #
            # gold_opinion_num_length1, pred_opinion_num_length1, correct_opinion_num_length1 = self.num_4_eval(
            #     gold_opinion_length1, pred_opinion_length1, gold_opinion_num_length1, pred_opinion_num_length1,
            #     correct_opinion_num_length1)
            # gold_opinion_num_length2, pred_opinion_num_length2, correct_opinion_num_length2 = self.num_4_eval(
            #     gold_opinion_length2, pred_opinion_length2, gold_opinion_num_length2, pred_opinion_num_length2,
            #     correct_opinion_num_length2)
            # gold_opinion_num_length3, pred_opinion_num_length3, correct_opinion_num_length3 = self.num_4_eval(
            #     gold_opinion_length3, pred_opinion_length3, gold_opinion_num_length3, pred_opinion_num_length3,
            #     correct_opinion_num_length3)
            # gold_opinion_num_length4, pred_opinion_num_length4, correct_opinion_num_length4 = self.num_4_eval(
            #     gold_opinion_length4, pred_opinion_length4, gold_opinion_num_length4, pred_opinion_num_length4,
            #     correct_opinion_num_length4)
            # gold_opinion_num_length5, pred_opinion_num_length5, correct_opinion_num_length5 = self.num_4_eval(
            #     gold_opinion_length5, pred_opinion_length5, gold_opinion_num_length5, pred_opinion_num_length5,
            #     correct_opinion_num_length5)
            # assert gold_opinion_num_length1+gold_opinion_num_length2+ gold_opinion_num_length3+gold_opinion_num_length4+gold_opinion_num_length5 == gold_opinion_num



            # F1 = open(self.args.dataset + 'aspect.json', 'w', encoding='utf-8')
            # json.dump(aspect_text, F1, ensure_ascii=False, indent=4)
            # F1.close()
            #
            # F2 = open(self.args.dataset + 'opinion.json', 'w', encoding='utf-8')
            # json.dump(opinion_text, F2, ensure_ascii=False, indent=4)
            # F2.close()

        # aspect_result_length1 = self.P_R_F1(gold_aspect_num_length1, pred_aspect_num_length1, correct_aspect_num_length1)
        # aspect_result_length2 = self.P_R_F1(gold_aspect_num_length2, pred_aspect_num_length2,
        #                                     correct_aspect_num_length2)
        # aspect_result_length3 = self.P_R_F1(gold_aspect_num_length3, pred_aspect_num_length3,
        #                                     correct_aspect_num_length3)
        # aspect_result_length4 = self.P_R_F1(gold_aspect_num_length4, pred_aspect_num_length4,
        #                                     correct_aspect_num_length4)
        # aspect_result_length5 = self.P_R_F1(gold_aspect_num_length5, pred_aspect_num_length5,
        #                                     correct_aspect_num_length5)

        # print('one token aspect precision:', aspect_result_length1[0], "one token aspect recall: ",
        #       aspect_result_length1[1], "one token aspect f1: ",
        #       aspect_result_length1[2])
        # print('two token aspect precision:', aspect_result_length2[0], "two token aspect recall: ",
        #       aspect_result_length2[1], "two token aspect f1: ",
        #       aspect_result_length2[2])
        # print('three token aspect precision:', aspect_result_length3[0], "three token aspect recall: ",
        #       aspect_result_length3[1], "three token aspect f1: ",
        #       aspect_result_length3[2])
        # print('4 token aspect precision:', aspect_result_length4[0], "4 token aspect recall: ",
        #       aspect_result_length4[1], "4 token aspect f1: ",
        #       aspect_result_length4[2])
        # print('5 token aspect precision:', aspect_result_length5[0], "5 token aspect recall: ",
        #       aspect_result_length5[1], "5 token aspect f1: ",
        #       aspect_result_length5[2])
        #
        # opinion_result_length1 = self.P_R_F1(gold_opinion_num_length1, pred_opinion_num_length1, correct_opinion_num_length1)
        # opinion_result_length2 = self.P_R_F1(gold_opinion_num_length2, pred_opinion_num_length2, correct_opinion_num_length2)
        # opinion_result_length3 = self.P_R_F1(gold_opinion_num_length3, pred_opinion_num_length3, correct_opinion_num_length3)
        # opinion_result_length4 = self.P_R_F1(gold_opinion_num_length4, pred_opinion_num_length4, correct_opinion_num_length4)
        # opinion_result_length5 = self.P_R_F1(gold_opinion_num_length5, pred_opinion_num_length5, correct_opinion_num_length5)
        #
        # print('one token opinion precision:', opinion_result_length1[0], "one token opinion recall: ", opinion_result_length1[1], "one token opinion f1: ",
        #       opinion_result_length1[2])
        # print('two token opinion precision:', opinion_result_length2[0], "two token opinion recall: ", opinion_result_length2[1], "two token opinion f1: ",
        #       opinion_result_length2[2])
        # print('three token opinion precision:', opinion_result_length3[0], "three token opinion recall: ", opinion_result_length3[1], "three token opinion f1: ",
        #       opinion_result_length3[2])
        # print('4 token opinion precision:', opinion_result_length4[0], "4 token opinion recall: ", opinion_result_length4[1], "4 token opinion f1: ",
        #       opinion_result_length4[2])
        # print('5 token opinion precision:', opinion_result_length5[0], "5 token opinion recall: ",
        #       opinion_result_length5[1], "5 token opinion f1: ",
        #       opinion_result_length5[2])


        # aspect_result = self.P_R_F1(gold_aspect_num, pred_aspect_num, correct_aspect_num)
        # opinion_result = self.P_R_F1(gold_opinion_num, pred_opinion_num, correct_opinion_num)
        # apce_result = self.P_R_F1(gold_apce_num, pred_apce_num, correct_apce_num)
        # pair_result = self.P_R_F1(gold_pairs_num, pred_pairs_num, correct_pairs_num)

        return quad_result,pair_result,low_standard,triples_result

    def find_token(self, bert_tokens, span):
        bert_tokens_4_span = bert_tokens[span[1]:span[2]+1]
        sub = ''
        for i, tokens in enumerate(bert_tokens_4_span):
            if i == 0:
                sub = tokens
            elif '##' in tokens:
                sub = sub + tokens.lstrip("##")
            else:
                sub = sub +" "+ tokens
        return sub

    def gold_token(self, tokens):
        sub = ''
        for i, token in enumerate(tokens):
            if i == 0:
                sub = token
            elif '##' in token:
                sub = sub + token.lstrip("##")
            else:
                sub = sub +" "+ token
        return sub

    def find_aspect_sentiment(self, sentence_index, bert_spans, span, aspect_sentiment, aspect_sentiment_logit):
        # span = [span[1], span[2], ]
        bert_span_index = [i for i,x in enumerate(bert_spans) if span[1] == x[0] and span[2] == x[1]]
        assert len(bert_span_index) == 1
        bert_span_index = bert_span_index[0]
        sentiment_index = aspect_sentiment[sentence_index][bert_span_index]
        # sentiment = id4sentiment[aspect_sentiment[sentence_index][bert_span_index]]
        sentiment = id4validity[aspect_sentiment[sentence_index][bert_span_index]]
        sentiment_logit = aspect_sentiment_logit[sentence_index][bert_span_index][sentiment_index]
        # all_sentiment_logit = sum(aspect_sentiment_logit[sentence_index][bert_span_index])
        # sentiment_precent = sentiment_logit / all_sentiment_logit
        # return sentiment, sentiment_precent

        return sentiment, sentiment_logit

    def find_aspect_category(self, sentence_index, bert_spans, span, aspect_category,j):
        # span = [span[1], span[2], ]
        _,id2category = get_categories(self.args)
        bert_span_index = [i for i,x in enumerate(bert_spans) if span[1] == x[0] and span[2] == x[1]]
        if j<len(bert_span_index):
            bert_span_index = bert_span_index[j]
            category = id2category[aspect_category[sentence_index][bert_span_index]]
        # sentiment_logit = aspect_sentiment_logit[sentence_index][bert_span_index][sentiment_index]
        # all_sentiment_logit = sum(aspect_sentiment_logit[sentence_index][bert_span_index])
        # sentiment_precent = sentiment_logit / all_sentiment_logit
        # return sentiment, sentiment_precent
            return category
        else:
            return " "
    def find_opinion_sentiment(self, sentence_index, opinion_index, bert_spans, span, opinion_sentiment,
                               opinion_sentiment_logit,aspect_logits,index,is_aspect=True):
        bert_span_index = [i for i, x in enumerate(bert_spans) if span[1] == x[0] and span[2] == x[1]]
        # if bert_span_index == 0:
        # assert len(bert_span_index) == 1
        if(len(bert_span_index)>1 and is_aspect==False):
            bert_span_index = bert_span_index[1]
        else:
            bert_span_index = bert_span_index[0]
        sentiment_index = opinion_sentiment[sentence_index][opinion_index][bert_span_index]
        sentiment = id4sentiment[opinion_sentiment[sentence_index][opinion_index][bert_span_index]]
        sentiment_logit = opinion_sentiment_logit[sentence_index][opinion_index][bert_span_index][sentiment_index]
        aspect_logits = aspect_logits[index][1]
        return sentiment, sentiment_logit,aspect_logits

    # 使用原始数据的代码
    def find_gold_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list,pair_list = [],[]
        aspect_list,opinion_list,apce_list = [],[],[]
        triples = self.gold_instances[sentence_index]['triples']
        for keys in triples:
            aspect, opinion = keys.split('|')
            aspect_tokens = []
            for aspect_token in aspect.split( ):
                token = self.tokenizer.tokenize(aspect_token)
                if self.args.span_generation == "CNN":
                    aspect_tokens.append(token[0])
                elif self.args.Only_token_head:
                    aspect_tokens.append(token[0])
                else:
                    aspect_tokens += token
            new_aspect = self.gold_token(aspect_tokens)

            opinion_tokens = []
            for opinion_token in opinion.split( ):
                token = self.tokenizer.tokenize(opinion_token)
                if self.args.span_generation == "CNN":
                    opinion_tokens.append(token[0])
                elif self.args.Only_token_head:
                    opinion_tokens.append(token[0])
                else:
                    opinion_tokens += token
            new_opinion = self.gold_token(opinion_tokens)

            sentiment = triples[keys][2]

            triples_list.append((new_aspect, new_opinion, sentiment.lower()))

            aspect_list.append((new_aspect))
            opinion_list.append((new_opinion))

            apce_list.append((new_aspect, sentiment))
            pair_list.append((new_aspect, new_opinion))
        return aspect_list, opinion_list, apce_list, pair_list, triples_list

    def find_gold_quad(self, sentence_index, bert_spans, bert_tokens):
        triples_list,pair_list,quad_list = [],[],[]
        aspect_list,opinion_list,apce_list,category_list = [],[],[],[]
        triples = self.gold_instances[sentence_index]['quad']
        for keys in triples:
            aspect, opinion = keys.split('|')
            aspect_tokens = []
            for aspect_token in aspect.split( ):
                token = self.tokenizer.tokenize(aspect_token)
                if self.args.span_generation == "CNN":
                    aspect_tokens.append(token[0])
                elif self.args.Only_token_head:
                    aspect_tokens.append(token[0])
                else:
                    aspect_tokens += token
            if(aspect_tokens == []):
                new_aspect = '[CLS]'
            else:
                new_aspect = self.gold_token(aspect_tokens)

            opinion_tokens = []
            for opinion_token in opinion.split( ):
                token = self.tokenizer.tokenize(opinion_token)
                if self.args.span_generation == "CNN":
                    opinion_tokens.append(token[0])
                elif self.args.Only_token_head:
                    opinion_tokens.append(token[0])
                else:
                    opinion_tokens += token
            if (opinion_tokens == []):
                new_opinion = '[CLS]'
            else:
                new_opinion = self.gold_token(opinion_tokens)
            # new_opinion = self.gold_token(opinion_tokens)

            sentiment = triples[keys][3]
            category = triples[keys][2]
            triples_list.append((new_aspect, new_opinion,sentiment.lower()))
            quad_list.append((new_aspect, new_opinion, category,sentiment.lower()))
            aspect_list.append((new_aspect))
            opinion_list.append((new_opinion))
            category_list.append((category))
            apce_list.append((new_aspect, sentiment))
            pair_list.append((new_aspect, new_opinion))
        return aspect_list, opinion_list, apce_list, pair_list, triples_list,quad_list,category_list


    def find_pred_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list, pair_list, span_list = [], [], []
        aspect_list, pred_opinion_list, apce_list = [], [], []
        pred_aspect_span = self.pred_aspect[sentence_index]
        # 去除重复的aspect
        new_aspect_span = []
        for i, pred_aspect in enumerate(pred_aspect_span):
            if len(new_aspect_span) == 0:
                new_aspect_span.append(pred_aspect)
            else:
                if pred_aspect[1] == new_aspect_span[-1][1]:
                    new_aspect_span[-1] = pred_aspect
                else:
                    new_aspect_span.append(pred_aspect)
        for j, pred_aspect in enumerate(new_aspect_span):
            if pred_aspect[1] == -1:
                aspect = "CLS"
            else:
                aspect = self.find_token(bert_tokens, pred_aspect)
                aspect_span_output = [pred_aspect[1], pred_aspect[2]+1]
                aspect_sentiment, aspect_sentiment_logit = self.find_aspect_category(sentence_index, bert_spans,
                                                                                      pred_aspect,
                                                                                      self.pred_aspect_category,
                                                                                      )
            aspect_list.append(aspect)

            opinion_list = []
            for opinion_index in list(np.where(np.array(self.pred_opinion[sentence_index][j]) != 0)[0]):
                opinion_list.append(opinion_index)
            opinion_spans = []
            for opinion_index in opinion_list:
                if opinion_index < len(bert_spans):
                    if opinion_index == 0:
                        opinion_spans.append([0,0,0])
                    else:
                        opinion_spans.append(bert_spans[opinion_index])
                else:
                    continue
            new_opinion_spans = []

            for i, pred_opinion in enumerate(opinion_spans):
                if len(new_opinion_spans) == 0:
                    new_opinion_spans.append(pred_opinion)
                else:
                    if pred_opinion[1] == new_opinion_spans[-1][1]:
                        new_opinion_spans[-1] = pred_opinion
                    else:
                        new_opinion_spans.append(pred_opinion)

            for opinion_span in new_opinion_spans:
                opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                opinion_span_output = [opinion_span[1], opinion_span[2]+1]
                if(opinion_span[2] == 0):
                    opinion = 'CLS'
                else:
                    opinion = self.find_token(bert_tokens, opinion_span)
                opinion_sentiment, opinion_sentiment_logit = self.find_opinion_sentiment(sentence_index, j, bert_spans,
                                                                                         opinion_span,
                                                                                         self.pred_opinion,
                                                                                         self.pred_opinion_sentiment_logit)
                # 筛选情感  弃用
                # if opinion_sentiment_logit > aspect_sentiment_logit:
                #     sentiment = opinion_sentiment
                # else:
                #     sentiment = aspect_sentiment

                pred_opinion_list.append(opinion)
                apce_list.append((aspect, opinion_sentiment))
                triples_list.append((aspect, opinion, opinion_sentiment))
                pair_list.append((aspect, opinion))
                span_list.append((aspect_span_output, opinion_span_output, opinion_sentiment_logit))
        return aspect_list, pred_opinion_list, apce_list, pair_list, triples_list, span_list


    def find_pred_quad(self, sentence_index, bert_spans, bert_tokens):
        triples_list, pair_list, span_list = [], [], []
        aspect_list, pred_opinion_list, apce_list = [], [], []
        category_list = []
        quad_list = []
        pred_aspect_span = self.pred_aspect[sentence_index]

        pred_aspect_logits = self.pred_aspect_sentiment_logit[sentence_index]


        asp_opi_imp = self.imp_asp_opi[sentence_index]
        opi_asp_imp = self.imp_opi_asp[sentence_index]
        _,id2category = get_categories(self.args)
        # pred_category_span = self.pred_aspect_category[sentence_index]
        # 去除重复的aspect
        new_aspect_span = []
        for i, pred_aspect in enumerate(pred_aspect_span):
            if len(new_aspect_span) == 0:
                new_aspect_span.append(pred_aspect)
            else:
                if pred_aspect[1] == new_aspect_span[-1][1]:
                    new_aspect_span[-1] = pred_aspect
                else:
                    new_aspect_span.append(pred_aspect)
        for j, pred_aspect in enumerate(new_aspect_span):
            aspect_index = pred_aspect[3]
            if pred_aspect[1] == 0:
                aspect = "[CLS]"
            else:
                aspect = self.find_token(bert_tokens, pred_aspect)
            aspect_span_output = [pred_aspect[1], pred_aspect[2]+1]#隐含情况：【0，1】


            aspect_list.append(aspect)
            # category_list.append(category)
            opinion_list = []
            for opinion_index in list(np.where(np.array(self.pred_opinion[sentence_index][j]) != 3)[0]):
                opinion_list.append(opinion_index)
            opinion_spans = []
            for opinion_index in opinion_list:
                if opinion_index < len(bert_spans):
                    if opinion_index == len(bert_spans)-1:
                        opinion_spans.append([0,0,0])
                    else:
                        opinion_spans.append(bert_spans[opinion_index])
                else:
                    continue
            new_opinion_spans = []

            for i, pred_opinion in enumerate(opinion_spans):
                if len(new_opinion_spans) == 0:
                    new_opinion_spans.append(pred_opinion)
                else:
                    if pred_opinion[0] == new_opinion_spans[-1][0]:
                        new_opinion_spans[-1] = pred_opinion
                    else:
                        new_opinion_spans.append(pred_opinion)

            for opinion_span in new_opinion_spans:
                opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                opinion_span_output = [opinion_span[1], opinion_span[2]+1]
                if (opinion_span[2] == 0):
                    opinion = '[CLS]'
                else:
                    opinion = self.find_token(bert_tokens, opinion_span)
                opinion_sentiment, opinion_sentiment_logit,aspect_logits = self.find_opinion_sentiment(sentence_index, j, bert_spans,
                                                                                         opinion_span,
                                                                                         self.pred_opinion,
                                                                                         self.pred_opinion_sentiment_logit,
                                                                                         pred_aspect_logits,
                                                                                         aspect_index,
                                                                                         False
                                                                                         )

                # 筛选情感  弃用
                # if opinion_sentiment_logit > aspect_sentiment_logit:
                #     sentiment = opinion_sentiment
                # else:
                #     sentiment = aspect_sentiment
                # if opinion_sentiment_logit*aspect_logits <= self.args.filter_a:
                #     continue
                pred_opinion_list.append(opinion)
                apce_list.append((aspect, opinion_sentiment))
                triples_list.append((aspect, opinion, opinion_sentiment))
                pair_list.append((aspect, opinion,opinion_sentiment_logit*aspect_logits))
                span_list.append((aspect_span_output, opinion_span_output, opinion_sentiment_logit))
                # quad_list.append((aspect,opinion,category,opinion_sentiment))

        quad_list1 = []
        quad_list2 = []
        for m,imp_asp_quad in enumerate(asp_opi_imp):
            aspect_span = [0,imp_asp_quad[1][0].int(),imp_asp_quad[1][0].int()]
            opinion_span = [0,imp_asp_quad[2][0].int(),imp_asp_quad[2][1].int()]
            aspect = self.find_token(bert_tokens,aspect_span)
            opinion = self.find_token(bert_tokens,opinion_span)
            category = id2category[imp_asp_quad[3]]
            sentiment = id4sentiment[imp_asp_quad[4]]
            quad_list1.append((aspect,opinion,category,sentiment))

        for m,imp_opi_quad in enumerate(opi_asp_imp):
            aspect_span = [0,imp_opi_quad[1][0].int(),imp_opi_quad[1][0].int()]
            opinion_span = [0,imp_opi_quad[2][0].int(),imp_opi_quad[2][1].int()]
            aspect = self.find_token(bert_tokens,aspect_span)
            opinion = self.find_token(bert_tokens,opinion_span)
            category = id2category[imp_opi_quad[3]]
            sentiment = id4sentiment[imp_opi_quad[4]]
            quad_list2.append((aspect,opinion,category,sentiment))
        quad_list = list(set(quad_list1) | set(quad_list2))
        '''
        if len(pred_category_span)>0:
            for i,a_o_c in enumerate(pred_category_span):
                aspect_span = bert_spans[a_o_c[0]]
                aspect_span = (aspect_span[2], aspect_span[0], aspect_span[1])

                opinion_span = bert_spans[int(a_o_c[1])]
                opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                aspect = self.find_token(bert_tokens,aspect_span)
                opinion = self.find_token(bert_tokens,opinion_span)
                for j,triple in enumerate(triples_list):
                    if triple[0]==aspect and triple[1] == opinion:
                        quad_list.append((aspect,opinion,id2category[a_o_c[2]],triple[2]))
        '''
        return aspect_list, pred_opinion_list, apce_list, pair_list, triples_list, span_list,quad_list,category_list
    def find_pred_reverse_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list, pair_list, span_list = [], [], []
        opinion_list, pred_aspect_list, apce_list = [], [], []
        pred_opinion_span = self.reverse_pred_opinon[sentence_index]

        reverse_opinion_logits = self.reverse_pred_opinon_sentiment_logit[sentence_index]


        new_opinion_span = []
        for i, pred_opinion in enumerate(pred_opinion_span):
            if len(new_opinion_span) == 0:
                new_opinion_span.append(pred_opinion)
            else:
                '''取长操作，重叠的实体取更长的部分'''
                if pred_opinion[1] == new_opinion_span[-1][1]:
                    new_opinion_span[-1] = pred_opinion
                else:
                    new_opinion_span.append(pred_opinion)
        for j, pred_opinion in enumerate(new_opinion_span):
            opinion_index = pred_opinion[3]
            opinion = self.find_token(bert_tokens, pred_opinion)
            opinion_span_output = [pred_opinion[1], pred_opinion[2] + 1]
            # opinion_sentiment, opinion_sentiment_precent = self.find_aspect_sentiment(sentence_index,
            #                                                                         bert_spans,
            #                                                                         pred_opinion,
            #                                                                         self.reverse_pred_opinon_sentiment,
            #                                                                         self.reverse_pred_opinon_sentiment_logit)
            opinion_list.append((opinion))
            aspect_list = []
            for aspect_index in list(np.where(np.array(self.reverse_pred_aspect[sentence_index][j]) != 3)[0]):
                aspect_list.append(aspect_index)
            aspect_spans = []
            for aspect_index in aspect_list:
                if aspect_index < len(bert_spans):
                    aspect_spans.append(bert_spans[aspect_index])
                else: continue
            new_aspect_spans = []
            '''同样的开头，选择更长的实体'''
            for i, pred_aspect in enumerate(aspect_spans):
                if len(new_aspect_spans) == 0:
                    new_aspect_spans.append(pred_aspect)
                else:

                    if pred_aspect[1] == new_aspect_spans[-1][1]:
                        new_aspect_spans[-1] = pred_aspect
                    else:
                        new_aspect_spans.append(pred_aspect)
            for aspect_span in new_aspect_spans:
                aspect_span = (aspect_span[2], aspect_span[0], aspect_span[1])
                aspect_span_output = [aspect_span[1], aspect_span[2] + 1]
                aspect = self.find_token(bert_tokens, aspect_span)
                aspect_sentiment, aspect_sentiment_precent,opinion_logits = self.find_opinion_sentiment(sentence_index, j,
                                                                                       bert_spans, aspect_span,
                                                                                       self.reverse_pred_aspect,
                                                                                       self.reverse_pred_aspect_sentiment_logit,
                                                                                         reverse_opinion_logits,
                                                                                         opinion_index)
                # if opinion_sentiment_precent > aspect_sentiment_precent:
                #     sentiment = opinion_sentiment
                # else:
                #     sentiment = aspect_sentiment
                # if aspect_sentiment_precent*opinion_logits <= self.args.filter_a:
                #     continue
                pred_aspect_list.append((aspect))
                apce_list.append((aspect, aspect_sentiment))
                triples_list.append((aspect, opinion, aspect_sentiment))
                pair_list.append((aspect, opinion,aspect_sentiment_precent*opinion_logits))
                span_list.append((aspect_span_output, opinion_span_output, aspect_sentiment_precent))
        return pred_aspect_list, opinion_list, apce_list, pair_list, triples_list, span_list

if __name__ == '__main__':
    test1 = ('boot time', 'fast', 'pos')
    test = ('boot time', 'boot')
    test2 = ('Boot time', 'fast', 'pos')
    set1  = set(test1) & set(test2)
    print(set(test))