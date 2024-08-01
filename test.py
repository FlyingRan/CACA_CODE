import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import json
# 初始化所需资源
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# 获取停用词和期望的词性标签
stop_words = set(stopwords.words('english'))
stop_words = stop_words - {'not','out'}
expected_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS','RB','RBR','RBS','VB'}  # 名词和形容词的词性标签
# ,'VBN','VBP','VBD','VBG','VBZ','IN'
special_symbols = set(string.punctuation)  # 特殊符号集合

# with open('restaurantquad_num_all.json', 'r',encoding='utf-8') as file:
#     data_json = json.load(file)


# 过滤条件函数
def filter_first_word_stopword_special_symbols(span, stop_words,special_symbols):
    # 过滤掉停用词在第一位的跨度
    return span[0] not in special_symbols and span[0] not in stop_words

def has_sufficient_meaningful_words(span, stop_words, min_meaningful_ratio=0.5):
    # 确保跨度中至少有一定比例的单词不是停用词
    meaningful_words = [word for word in span if word.lower() not in stop_words]
    return len(meaningful_words) / len(span) >= min_meaningful_ratio

def contains_expected_pos_tags(span, expected_pos_tags):
    # 确保跨度中包含名词或形容词
    pos_tags = nltk.pos_tag(span)
    return any(tag in expected_pos_tags for word, tag in pos_tags)



def is_grammatically_complete(span_text):
    doc = nlp(span_text)
    for token in doc:
        if token.dep_ in {'punct', 'det', 'prep'} and token.head == token:
            return False
        if token.dep_ == 'prep' and not any(child.dep_ == 'pobj' for child in token.children):
            return False
    for token in doc:
        if token.dep_ == 'ROOT':
            return True
    return False
# 综合过滤函数

def is_pos_span(span):
    # 使用spaCy进行依存关系解析
    span_text = ' '.join(span)
    doc = nlp(span_text)
    # 检查依存关系的合理性
    for token in doc:
        if token.dep_ in {'punct', 'det', 'prep'} and token.head == token:
            return False
        if token.dep_ == 'prep' and not any(child.dep_ == 'pobj' for child in token.children):
            return False
    return True

def is_valid_span(span):
    # if span == "":
    #     return True
    # if not filter_first_word_stopword_special_symbols(span, stop_words,special_symbols):
    #     return False
    # if not has_sufficient_meaningful_words(span, stop_words):
    #     return False
    # if not contains_expected_pos_tags(span, expected_pos_tags):
    #     return False
    # span_text = ' '.join(span)
    # if not is_grammatically_complete(span_text) and len(span) > 1:
    #     return False
    # if not is_pos_span(span):
    #     return False
    return True

def filter_invalid_spans(spans):
    return [span for span in spans if is_valid_span(span)]

def generate_spans(text, max_span_length=8):
    words = text.split()  # 将文本拆分成单词列表
    spans = []  # 存储所有跨度

    # 遍历每个单词并生成跨度
    for i in range(len(words)):
        for j in range(1, max_span_length + 1):
            if i + j <= len(words):
                span = words[i:i + j]
                spans.append(span)

    return spans


'''
i=0
aspect_all_count = 0
opinion_all_count = 0
aspect_fil_count = 0
opinion_fil_count = 0
fil_gold_aspect = []
fil_gold_opinion = []
total_fil_span = 0
total_all_span = 0
for example in data_json:
    quad_list_gold = example.get('quad_list_gold', [])
    # quad_list_pred = example.get('quad_list_pred', [])
    sentence = example.get('sentence')
    spans = generate_spans(sentence)
    valid_spans = filter_invalid_spans(spans)
    total_fil_span += len(valid_spans)
    total_all_span += len(spans)
    i = i+1
    print("正在处理第{}条文本".format(i))
    for quad in quad_list_gold:
        aspect = quad[0]
        opinion = quad[1]

        if aspect != "[CLS]":
            aspect = quad[0][1:]
            aspect = aspect.replace(' ', '')
            aspect = aspect.replace('▁', ' ')
            aspect_all_count += 1
            aspect = aspect.split(' ')
            if aspect in valid_spans:
                aspect_fil_count +=1
            else:
                fil_gold_aspect.append(aspect)
        if opinion != "[CLS]":
            opinion_all_count += 1
            opinion = quad[1][1:]
            opinion = opinion.replace(' ', '')
            opinion = opinion.replace('▁', ' ')
            opinion = opinion.split(' ')
            if opinion in valid_spans:
                opinion_fil_count += 1
            else:
                fil_gold_opinion.append(opinion)


print("不应该被过滤掉的方面词",fil_gold_aspect)
print("方面词过滤后{}/{}".format(aspect_fil_count,aspect_all_count))
print("不应该被过滤掉的意见词",fil_gold_opinion)
print("意见词过滤后{}/{}".format(opinion_fil_count,opinion_all_count))
print("过滤后的跨度数量和原本的跨度数量，{}/{}".format(total_fil_span,total_all_span))
'''

#


# count_cls_in_aspect_opinion(data_json)

# # 分割成跨度（这里只是示例，实际使用中可能需要根据具体需求分割）
# # spans = [ ['amazing'],['accomodating' ] ,['die', 'for'], ['must'],['loved'],['enjoyed']]
# #
# #
# # # 过滤无效跨度
# # valid_spans = filter_invalid_spans(spans, stop_words, expected_pos_tags, special_symbols)
# #
# # for span in valid_spans:
# #     print(span)