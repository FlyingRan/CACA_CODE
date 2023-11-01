import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
# from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .Attention import Attention, Intermediate, Output, Dim_Four_Attention, masked_softmax,implict_ATT
from .data_BIO_loader import sentiment2id, validity2id
from allennlp.nn.util import batched_index_select, batched_span_select
import random
import math

def stage_2_features_generation1(bert_feature, attention_mask, spans, span_mask, spans_embedding, spans_embedding1,spans_aspect_tensor,
                                is_aspect,soft_prop= None,spans_opinion_tensor=None):
    # 对输入的aspect信息进行处理，去除掉无效的aspect span
    all_span_aspect_tensor = None
    all_span_opinion_tensor = None
    all_bert_embedding = None
    all_attention_mask = None
    all_spans_embedding = None
    all_spans_embedding1 = None
    all_span_mask = None
    all_left_tensor = None
    all_right_tensor = None
    all_prob_tensor = None
    spans_aspect_tensor_spilt = torch.chunk(spans_aspect_tensor, spans_aspect_tensor.shape[0], dim=0)
    flag = 0
    for i, spans_aspect_tensor_unspilt in enumerate(spans_aspect_tensor_spilt):
        test = spans_aspect_tensor_unspilt.squeeze()
        batch_num = spans_aspect_tensor_unspilt.squeeze(0)[0]
        last = torch.sum(span_mask[batch_num]).item() - 1
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        if (test[1] == test[2] == -1) or (test[1] == test[2] == 0):
            flag = 1
            # left = torch.mean(spans_embedding[batch_num, 1:last, :],dim=0)
            # right,_ = torch.max(spans_embedding[batch_num, 1:last, :],dim=0)
        else:
            span_index_start = torch.where(spans[batch_num, :, 0] == spans_aspect_tensor_unspilt.squeeze()[1],
                                           spans[batch_num, :, 1], torch.tensor(-1).type_as(spans))
            span_index_end = torch.where(span_index_start == spans_aspect_tensor_unspilt.squeeze()[2], span_index_start,
                                         torch.tensor(-1).type_as(spans))
            span_index = torch.nonzero((span_index_end > -1), as_tuple=False).squeeze(0)

            # left = torch.mean(spans_embedding[batch_num, 1:span_index, :], dim=0) if span_index != 1 else \
            # spans_embedding[batch_num][0]
            # right = torch.mean(spans_embedding[batch_num, span_index:last, :], dim=0) if span_index != last else \
            # spans_embedding[batch_num][last]
        # if min(span_index.shape) == 0:
        #     continue
        if spans_opinion_tensor is not None:
            spans_opinion_tensor_unspilt = spans_opinion_tensor[i,:].unsqueeze(0)
        if flag != 1:
            aspect_span_embedding_unspilt = spans_embedding[batch_num, span_index, :].unsqueeze(0)
        else:
            if is_aspect:
                aspect_span_embedding_unspilt = spans_embedding[batch_num, torch.tensor([0]), :].unsqueeze(0)
            else:
                aspect_span_embedding_unspilt = spans_embedding[batch_num,  torch.tensor([last]), :].unsqueeze(0)
            flag = 0

        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        spans_embedding_unspilt1 = spans_embedding1[batch_num,:,:].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)
        if soft_prop is not None:
            prob_tensor = soft_prop[batch_num].unsqueeze(0)
        if all_span_aspect_tensor is None:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = spans_opinion_tensor_unspilt
            all_span_aspect_tensor = aspect_span_embedding_unspilt
            all_bert_embedding = bert_feature_unspilt
            all_attention_mask = attention_mask_unspilt
            all_spans_embedding = spans_embedding_unspilt
            all_spans_embedding1 = spans_embedding_unspilt1
            all_span_mask = span_mask_unspilt
            # all_left_tensor = left
            # all_right_tensor = right
            if soft_prop is not None:
                all_prob_tensor = prob_tensor
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)
            num_dims = len(all_span_aspect_tensor.shape)
            if soft_prop is not None:
                all_prob_tensor = torch.cat((all_prob_tensor,prob_tensor),dim=0)
            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            # all_left_tensor = torch.cat((all_left_tensor, left), dim=0)
            # all_right_tensor = torch.cat((all_right_tensor, right), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_spans_embedding1 = torch.cat((all_spans_embedding1, spans_embedding_unspilt1), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
           all_spans_embedding, all_spans_embedding1,all_span_mask,all_left_tensor,all_right_tensor,all_prob_tensor
def stage_2_features_generation(bert_feature, attention_mask, spans, span_mask, spans_embedding, spans_aspect_tensor,
                                is_aspect,soft_prop= None,spans_opinion_tensor=None):
    # 对输入的aspect信息进行处理，去除掉无效的aspect span
    all_span_aspect_tensor = None
    all_span_opinion_tensor = None
    all_bert_embedding = None
    all_attention_mask = None
    all_spans_embedding = None
    all_span_mask = None
    all_left_tensor = None
    all_right_tensor = None
    all_prob_tensor = None
    spans_aspect_tensor_spilt = torch.chunk(spans_aspect_tensor, spans_aspect_tensor.shape[0], dim=0)
    flag = 0
    for i, spans_aspect_tensor_unspilt in enumerate(spans_aspect_tensor_spilt):
        test = spans_aspect_tensor_unspilt.squeeze()
        batch_num = spans_aspect_tensor_unspilt.squeeze(0)[0]
        last = torch.sum(span_mask[batch_num]).item() - 1
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        if (test[1] == test[2] == -1) or (test[1] == test[2] == 0):
            flag = 1
            # left = torch.mean(spans_embedding[batch_num, 1:last, :],dim=0)
            # right,_ = torch.max(spans_embedding[batch_num, 1:last, :],dim=0)
        else:
            span_index_start = torch.where(spans[batch_num, :, 0] == spans_aspect_tensor_unspilt.squeeze()[1],
                                           spans[batch_num, :, 1], torch.tensor(-1).type_as(spans))
            span_index_end = torch.where(span_index_start == spans_aspect_tensor_unspilt.squeeze()[2], span_index_start,
                                         torch.tensor(-1).type_as(spans))
            span_index = torch.nonzero((span_index_end > -1), as_tuple=False).squeeze(0)

            # left = torch.mean(spans_embedding[batch_num, 1:span_index, :], dim=0) if span_index != 1 else \
            # spans_embedding[batch_num][0]
            # right = torch.mean(spans_embedding[batch_num, span_index:last, :], dim=0) if span_index != last else \
            # spans_embedding[batch_num][last]
        # if min(span_index.shape) == 0:
        #     continue
        if spans_opinion_tensor is not None:
            spans_opinion_tensor_unspilt = spans_opinion_tensor[i,:].unsqueeze(0)
        if flag != 1:
            aspect_span_embedding_unspilt = spans_embedding[batch_num, span_index, :].unsqueeze(0)
        else:
            if is_aspect:
                aspect_span_embedding_unspilt = spans_embedding[batch_num, torch.tensor([0]), :].unsqueeze(0)
            else:
                aspect_span_embedding_unspilt = spans_embedding[batch_num,  torch.tensor([last]), :].unsqueeze(0)
            flag = 0

        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)
        if soft_prop is not None:
            prob_tensor = soft_prop[batch_num].unsqueeze(0)
        if all_span_aspect_tensor is None:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = spans_opinion_tensor_unspilt
            all_span_aspect_tensor = aspect_span_embedding_unspilt
            all_bert_embedding = bert_feature_unspilt
            all_attention_mask = attention_mask_unspilt
            all_spans_embedding = spans_embedding_unspilt
            all_span_mask = span_mask_unspilt
            # all_left_tensor = left
            # all_right_tensor = right
            if soft_prop is not None:
                all_prob_tensor = prob_tensor
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)
            num_dims = len(all_span_aspect_tensor.shape)
            if soft_prop is not None:
                all_prob_tensor = torch.cat((all_prob_tensor,prob_tensor),dim=0)
            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            # all_left_tensor = torch.cat((all_left_tensor, left), dim=0)
            # all_right_tensor = torch.cat((all_right_tensor, right), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
           all_spans_embedding, all_span_mask,all_left_tensor,all_right_tensor,all_prob_tensor

class Step_1_module(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_1_module, self).__init__()
        self.args = args
        self.intermediate = Intermediate(bert_config)
        self.output = Output(bert_config)

    def forward(self, spans_embedding):
        intermediate_output = self.intermediate(spans_embedding)
        layer_output = self.output(intermediate_output, spans_embedding)
        return layer_output, layer_output

class HierarchicalLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_layers):
        super(HierarchicalLSTMWithAttention, self).__init__()
        self.bottom_lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim, num_layers=num_layers, batch_first=True)
        self.top_lstm = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True)
        self.attention = nn.Linear(lstm_hidden_dim, 1)  # 注意力权重

    def forward(self, input_sequence):
        # 底层LSTM处理局部信息
        bottom_lstm_output, _ = self.bottom_lstm(input_sequence)

        # 顶层LSTM处理全局语义信息
        top_lstm_output, _ = self.top_lstm(bottom_lstm_output)

        # 使用注意力机制获取权重
        attention_weights = torch.softmax(self.attention(top_lstm_output), dim=1)

        # 使用注意力权重加权顶层LSTM输出
        attended_output = torch.sum(top_lstm_output * attention_weights, dim=1)

        return attended_output

class Step_1(torch.nn.Module):
    def feature_slice(self, features, mask, span_mask, sentence_length):
        cnn_span_generate_list = []
        for j, CNN_generation_model in enumerate(self.CNN_span_generation):
            bert_feature = features.permute(0, 2, 1)
            cnn_result = CNN_generation_model(bert_feature)
            cnn_span_generate_list.append(cnn_result)

        features_sliced_tensor = None
        features_mask_tensor = None
        for i in range(features.shape[0]):
            last_mask = torch.nonzero(mask[i, :])
            features_sliced = features[i,:last_mask.shape[0]][1:-1]
            for j in range(self.args.max_span_length -1):
                if last_mask.shape[0] - 2 > j:
                    # test = cnn_span_generate_list[j].permute(0, 2, 1)
                    cnn_feature = cnn_span_generate_list[j].permute(0, 2, 1)[i, 1:last_mask.shape[0] - (j+2), :]
                    features_sliced = torch.cat((features_sliced, cnn_feature), dim=0)
                else:
                    break
            pad_length = span_mask.shape[1] - features_sliced.shape[0]
            spans_mask_tensor = torch.full([1, features_sliced.shape[0]], 1, dtype=torch.long).to(self.args.device)
            if pad_length > 0:
                pad = torch.full([pad_length, self.args.bert_feature_dim], 0, dtype=torch.long).to(self.args.device)
                features_sliced = torch.cat((features_sliced, pad),dim=0)
                mask_pad = torch.full([1, pad_length], 0, dtype=torch.long).to(self.args.device)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad),dim=1)
            if features_sliced_tensor is None:
                features_sliced_tensor = features_sliced.unsqueeze(0)
                features_mask_tensor = spans_mask_tensor
            else:
                features_sliced_tensor = torch.cat((features_sliced_tensor, features_sliced.unsqueeze(0)), dim=0).to(self.args.device)
                features_mask_tensor = torch.cat((features_mask_tensor, spans_mask_tensor), dim=0).to(self.args.device)

        return features_sliced_tensor, features_mask_tensor

    def __init__(self, args, bert_config):
        super(Step_1, self).__init__()

        categories_class = 13 if args.dataset == "restaurant" else 121

        self.args = args
        self.bert_config = bert_config
        self.dropout_output = torch.nn.Dropout(args.drop_out)
        self.forward_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)

        # self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.reverse_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])

        self.category_classifier = nn.Sequential(
            nn.Linear(768 * 2, categories_class)
        )

        # self.aspect_imp_decoders = nn.ModuleList(
        #     [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        #
        # self.opinion_imp_decoders = nn.ModuleList(
        #     [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])

        self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)
        # self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        # self.categories_classification_opinion = nn.Linear(args.bert_feature_dim, len(categories))
        self.ATT_attentions = nn.ModuleList(
            [Dim_Four_Block(args, self.bert_config) for _ in range(max(1, args.ATT_SPAN_block_num - 1))])
        self.multhead_asp = nn.MultiheadAttention(embed_dim=args.bert_feature_dim, num_heads=4)

        self.multhead_opi = nn.MultiheadAttention(embed_dim=args.bert_feature_dim, num_heads=4)

        self.imp_asp_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim, 2)
        )
        self.imp_opi_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim, 2)
        )

        # self.compess_projection = nn.Sequential(nn.Linear(args.bert_feature_dim, 1), nn.ReLU(),
        #                                             nn.Dropout(args.drop_out))

        self.opinion_lstm = HierarchicalLSTMWithAttention(args.bert_feature_dim,128,2)
        self.aspect_lstm = HierarchicalLSTMWithAttention(args.bert_feature_dim, 128, 2)

        # self._densenet = nn.Sequential(nn.Linear(args.bert_feature_dim*4, args.bert_feature_dim),
        #                                nn.Tanh())
        self._densenet = nn.Linear(args.bert_feature_dim*4, args.bert_feature_dim)
    def forward(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor, pooler_output,sentence_length):
        # 调用 span_generator 方法生成 spans_embedding 和 features_mask_tensor 两个变量。
        # 其中 spans_embedding 表示每个 span 的嵌入表示，features_mask_tensor 表示每个 span 是否被标记为有效的情感单元。
        # test = self.opinion_lstm(input_bert_features)
        spans_embedding, features_mask_tensor = self.span_generator(input_bert_features, attention_mask, spans,
                                                                    span_mask, related_spans_tensor,pooler_output, sentence_length)

        span_embedding_1 = torch.clone(spans_embedding)
        for forward_1_decoder in self.forward_1_decoders:
            forward_layer_output, forward_intermediate_output = forward_1_decoder(span_embedding_1)
            span_embedding_1 = forward_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 aspect 的预测结果，
        # 使用 sentiment_classification_aspect 层将其映射到具体的情感极性标签上，得到 class_logits_aspect。
        class_logits_aspect = self.sentiment_classification_aspect(span_embedding_1)
        # class_logits_category = self.categories_classification_opinion(span_embedding_1)
        # 复制 spans_embedding 得到 span_embedding_1 和 span_embedding_2 两个变量，
        # 然后分别将它们输入到 forward_1_decoders 和 reverse_1_decoders 中进行解码。
        span_embedding_2 = torch.clone(spans_embedding)
        for reverse_1_decoder in self.reverse_1_decoders:
            reverse_layer_output, reverse_intermediate_output = reverse_1_decoder(span_embedding_2)
            span_embedding_2 = reverse_layer_output
        # 同样用 sentiment_classification_opinion 层将其映射到具体的情感极性标签上，得到 class_logits_opinion。
        class_logits_opinion = self.sentiment_classification_opinion(span_embedding_2)

        span_embedding_3 = torch.clone(spans_embedding)
        span_embedding_4 = torch.clone(spans_embedding)

        # for aspect_imp_decoder in self.aspect_imp_decoders:
        #     aspect_imp_layer_output, aspect_imp_intermediate_output = aspect_imp_decoder(span_embedding_3)
        #     span_embedding_3 = aspect_imp_layer_output
        #
        # for opinion_imp_decoder in self.opinion_imp_decoders:
        #     opinion_imp_layer_output, opinion_imp_intermediate_output = opinion_imp_decoder(span_embedding_4)
        #     span_embedding_4 = opinion_imp_layer_output

        output1, _ = self.multhead_asp(span_embedding_3, span_embedding_3, span_embedding_3)
        output2, _ = self.multhead_opi(span_embedding_4, span_embedding_4, span_embedding_4)
        input_vector1 = output1[:,0,:].unsqueeze(1)
        input_vector2 = output2[range(span_embedding_4.shape[0]), torch.sum(span_mask, dim=-1) - 1].unsqueeze(1)
        input_vector2 = input_vector2.view(span_embedding_4.shape[0], -1)
        input_vector1 = input_vector1.view(span_embedding_3.shape[0], -1)

        # opinion_lst_rep = self.opinion_lstm(input_vector2)
        imp_aspect_exist = self.imp_asp_classifier(input_vector1)
        imp_opinion_exist = self.imp_opi_classifier(input_vector2)
        return class_logits_aspect, class_logits_opinion, spans_embedding, span_embedding_1, span_embedding_2, \
               features_mask_tensor,imp_aspect_exist,imp_opinion_exist

    def create_category_label(self,real_pair,pred_pair,real_category):
        final_label = []
        for pair in pred_pair:
            pair = pair.expand(real_pair.size(0), -1)
            # 使用逐元素比较和torch.where查找索引
            is_found = torch.all(pair == real_pair, dim=1)
            # 找到匹配的索引
            indices = torch.nonzero(is_found)
            if len(indices) > 0:
                for idx in range(len(indices)):
                    row_idx = indices[idx][0].item()
                    break
                final_label.append(int(real_category[0][row_idx]))
            else:
                final_label.append(0)
        final_label = torch.tensor(final_label).to(self.args.device)
        return final_label
    def cal_real(self,ao_pair,spans):
        real_pair = []
        ao_pair = torch.tensor(ao_pair)
        for span in ao_pair:
            span = span.clone().detach().to(self.args.device)
            aspect_indices = torch.nonzero(torch.all(torch.eq(spans, span[0]), dim=1))
            # 如果有匹配的索引，返回第一个匹配的索引；如果没有匹配的索引，返回0
            aspect_index = aspect_indices[0][0].item() if aspect_indices.numel() > 0 else 0
            opinion_indices = torch.nonzero(torch.all(torch.eq(spans, span[1]), dim=1))
            # 如果有匹配的索引，返回第一个匹配的索引；如果没有匹配的索引，返回0
            opinion_index = opinion_indices[0][0].item() if opinion_indices.numel() > 0 else 0
            real_pair.append([aspect_index,opinion_index])
        return real_pair
    def cartesian_pair(self,aspect_span,opinion_span):
        pred_pair = []
        for i in range(len(aspect_span)):
            for j in range(len(opinion_span)):
                pred_pair.append([aspect_span[i][0][1],opinion_span[j][0][1]])
        return pred_pair
    def cartesian_product(self,aspect_rep,opinion_rep):
        # 使用广播计算笛卡尔积并拼接
        expanded_aspect = aspect_rep.unsqueeze(1)
        expanded_opinion = opinion_rep.unsqueeze(0)

        expanded_aspect = expanded_aspect.repeat(1, opinion_rep.size(0), 1)
        expanded_opinion = expanded_opinion.repeat(aspect_rep.size(0), 1, 1)

        cartesian_product = torch.cat((expanded_aspect, expanded_opinion),dim=2)
        # 转换形状
        result = cartesian_product.reshape(aspect_rep.size(0)*opinion_rep.size(0), 768*2)
        return  result
    def span_generator(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor,pooler_output,
                       sentence_length):
        bert_feature = self.dropout_output(input_bert_features)
        features_mask_tensor = None
        if self.args.span_generation == "Average" or self.args.span_generation == "Max":
            # 如果使用全部span的bert信息：
            spans_num = spans.shape[1]# 有多少个span 包括了前后的【0，0，0】
            spans_width_start_end = spans[:, :, 0:2].view(spans.size(0), spans_num, -1)# 取所有span的前两位
            spans_width_start_end[0, -1] = spans_width_start_end[0, -2].clone()
            last_one_index = span_mask.flip(dims=(1,)).argmax(dim=1)
            last_one_index = span_mask.size(1) - 1 - last_one_index
            for i in range(len(span_mask)):
                last_index = int(last_one_index[i])
                spans_width_start_end[i, last_index][0] = spans_width_start_end[i, last_index-1][1] + 1
                spans_width_start_end[i, last_index][1] = spans_width_start_end[i, last_index-1][1] + 1
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))

            if self.args.span_generation == "Max":
                masked_representations = spans_width_start_end_embedding.masked_fill(~spans_width_start_end_mask,float('-inf'))
                spans_width_start_end_max = masked_representations.max(2)
                spans_embedding = spans_width_start_end_max[0]
            else:
                spans_width_start_end_mean = spans_width_start_end_embedding.mean(dim=2, keepdim=True).squeeze(-2)
                spans_embedding = spans_width_start_end_mean
            # for i in range(len(sentence_length)):
            #     spans_embedding[i][0] = pooler_output[i].clone().squeeze()
            #     spans_embedding[i][sentence_length[i][2]-1] = pooler_output[i].clone().squeeze()
        elif self.args.span_generation == "Start_end_minus_plus":
            spans_start = spans[:, :, 0].view(spans.size(0), -1)
            spans_end = spans[:, :, 1].view(spans.size(0), -1)
            last_one_index = span_mask.flip(dims=(1,)).argmax(dim=1)
            last_one_index = span_mask.size(1) - 1 - last_one_index
            for i in range(len(span_mask)):
                last_index = int(last_one_index[i])
                spans_start[i][last_index] = spans_end[i][last_index-1]+ 1
                spans_end[i][last_index] = spans_end[i][last_index-1]+ 1

            spans_start_embedding = batched_index_select(bert_feature, spans_start)
            spans_end_embedding = batched_index_select(bert_feature, spans_end)
            spans_embedding = torch.cat((spans_start_embedding,spans_end_embedding,spans_end_embedding-spans_start_embedding,spans_end_embedding+spans_start_embedding),dim=-1)
            spans_embedding = self._densenet(spans_embedding)
            # for i in range(len(sentence_length)):
            #     spans_embedding[i][0] = pooler_output[i].squeeze()
            #     spans_embedding[i][sentence_length[i][2]-1] = pooler_output[i].squeeze()
        elif self.args.span_generation == "Start_end":
            # 如果使用span区域大小进行embedding
            spans_start = spans[:, :, 0].view(spans.size(0), -1)
            spans_start_embedding = batched_index_select(bert_feature, spans_start)
            spans_end = spans[:, :, 1].view(spans.size(0), -1)
            spans_end_embedding = batched_index_select(bert_feature, spans_end)

            spans_width = spans[:, :, 2].view(spans.size(0), -1)
            spans_width_embedding = self.step_1_embedding4width(spans_width)
            spans_embedding = torch.cat((spans_start_embedding, spans_width_embedding, spans_end_embedding), dim=-1)  # 预留可修改部分
            # spans_embedding_dict = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
            spans_embedding_dict = self.step_1_linear4width(spans_embedding)
            spans_embedding = spans_embedding_dict
        elif self.args.span_generation == "CNN":
            feature_slice, features_mask_tensor = self.feature_slice(bert_feature, attention_mask, span_mask,
                                                                     sentence_length)
            spans_embedding = feature_slice
        elif self.args.span_generation == "ATT":
            spans_width_start_end = spans[:, :, 0:2].view(spans.shape[0], spans.shape[1], -1)
            spans_width_start_end[0, -1] = spans_width_start_end[0, -2].clone()
            last_one_index = span_mask.flip(dims=(1,)).argmax(dim=1)
            last_one_index = span_mask.size(1) - 1 - last_one_index
            for i in range(len(span_mask)):
                last_index = int(last_one_index[i])
                spans_width_start_end[i, last_index][0] = spans_width_start_end[i, last_index - 1][1] + 1
                spans_width_start_end[i, last_index][1] = spans_width_start_end[i, last_index - 1][1] + 1
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            span_sum_embdding = torch.sum(spans_width_start_end_embedding, dim=2).unsqueeze(2)
            for ATT_attention in self.ATT_attentions:
                ATT_layer_output, ATT_intermediate_output = ATT_attention(span_sum_embdding,
                                                                                      spans_width_start_end_mask,
                                                                                      spans_width_start_end_embedding)
                span_sum_embdding = ATT_layer_output
            spans_embedding = span_sum_embdding.squeeze()
        elif self.args.span_generation == "SE_ATT":
            spans_width_start_end = spans[:, :, 0:2].view(spans.shape[0], spans.shape[1], -1)
            spans_width_start_end[0, -1] = spans_width_start_end[0, -2].clone()

            last_one_index = span_mask.flip(dims=(1,)).argmax(dim=1)
            last_one_index = span_mask.size(1) - 1 - last_one_index
            for i in range(len(span_mask)):
                last_index = int(last_one_index[i])
                spans_width_start_end[i, last_index][0] = spans_width_start_end[i, last_index-1][1] + 1
                spans_width_start_end[i, last_index][1] = spans_width_start_end[i, last_index-1][1] + 1
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask_2 = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask_2, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))
            claim_self_att = self.compess_projection(spans_width_start_end_embedding).squeeze()
            claim_self_att = torch.sum(spans_width_start_end_embedding, dim=-1).squeeze()
            claim_rep = masked_softmax(claim_self_att, span_mask, spans_width_start_end_mask).unsqueeze(-1).transpose(2, 3)
            claim_rep = torch.matmul(claim_rep, spans_width_start_end_embedding)
            spans_embedding = claim_rep.squeeze()
        return spans_embedding, features_mask_tensor


class Dim_Four_Block(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Dim_Four_Block, self).__init__()
        self.args = args
        self.forward_attn = Dim_Four_Attention(bert_config)
        self.intermediate = Intermediate(bert_config)
        self.output = Output(bert_config)
    def forward(self, hidden_embedding, masks, encoder_embedding):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, hidden_dim, encoder_dim)
        masks = (~masks) * -1e9
        attention_masks = masks[:, :, None, None, :]
        cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                   encoder_hidden_states=encoder_embedding,
                                                   encoder_attention_mask=attention_masks)
        attention_output = cross_attention_output[0]
        attention_result = cross_attention_output[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_result



class Pointer_Block(torch.nn.Module):
    def __init__(self, args, bert_config, mask_for_encoder=True):
        super(Pointer_Block, self).__init__()
        self.args = args
        self.forward_attn = Attention(bert_config)
        # 创建 intermediate 层，用于对 attention_output 进行非线性变换，提高特征表达的能力
        self.intermediate = Intermediate(bert_config)
        # 创建 output 层，用于将经过 intermediate 层处理之后的特征映射到实际输出空间中
        self.output = Output(bert_config)
        self.mask_for_encoder = mask_for_encoder
        self.lstm = nn.LSTM(input_size=args.bert_feature_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.dense = nn.Linear(128, args.bert_feature_dim)
    def forward(self, hidden_embedding, masks, encoder_embedding):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, hidden_dim, encoder_dim)
        # 计算掩码，将 ~masks 乘以 -1e9，使得无意义位置对应的值趋近于负无穷
        lstm_outputs, (lstm_hidden, lstm_cell) = self.lstm(hidden_embedding)
        outputs = self.dense(lstm_outputs)
        # outputs, (lstm_hidden, lstm_cell) = self.lstm(hidden_embedding)

        masks = (1-masks) * -1e9
        # 根据掩码的维度信息确定 attention_masks 的形状

        if masks.dim() == 3:
            attention_masks = masks[:, None, :, :]
        elif masks.dim() == 2:
            if self.mask_for_encoder:
                attention_masks = masks[:, None, None, :]
            else:
                attention_masks = masks[:, None, :, None]
                #attention_masks = masks[:,:, None]
        if self.mask_for_encoder:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                       lstm_states = outputs,
                                                       encoder_hidden_states=encoder_embedding,
                                                       encoder_attention_mask=attention_masks)
        else:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                       lstm_states=outputs,
                                                       encoder_hidden_states=encoder_embedding,
                                                       attention_mask=attention_masks)
        # 获取 cross_attention_output 中的 attention_output 和 attention_result
        attention_output = cross_attention_output[0]
        attention_result = cross_attention_output[1:]
        # 对 attention_output 进行非线性变换，提高特征表达的能力
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 作为输入，进行投影映射，得到最终输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_result


class Step_2_forward1(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward1, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 用于将解码器输出的特征映射到具体的情感极性标签的数字 ID 上
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.reverse_decoder = Pointer_Block(args, self.bert_config, mask_for_encoder=False)
        self.output = Output(bert_config)
        self.dense = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)
        self.intermediate = Intermediate(bert_config)
    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor,opinion_prob,reverse_embedding):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output

        reverse_embedding,_ = self.reverse_decoder(reverse_embedding,
                                                   aspect_span_mask,
                                                   spans_aspect_tensor)

        reverse_embedding = self.intermediate(reverse_embedding)
        final_rep = self.output(reverse_embedding,aspect_spans_embedding)
        # 将最后一个解码器的输出结果作为情感极性关于 opinion 的预测结果，
        # 使用 opinion_docoder2class 层将其映射到具体的情感极性标签上，得到 opinion_class_logits。
        # input_rep = torch.cat((aspect_spans_embedding,all_left_tensor,all_right_tensor),dim=0)
        # opinion_prob = opinion_prob.unsqueeze(-1)
        # prob_embedding = self.dense(torch.mul(opinion_prob,aspect_spans_embedding))
        # final_embedding = torch.cat((aspect_spans_embedding, reverse_embedding), dim=-1)
        # final_embedding = self.dense(final_embedding)
        opinion_class_logits = self.opinion_docoder2class(final_rep)


        return opinion_class_logits

class Step_2_forward(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 用于将解码器输出的特征映射到具体的情感极性标签的数字 ID 上
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor,opinion_prob,reverse_embedding):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output
        opinion_class_logits = self.opinion_docoder2class(aspect_spans_embedding)


        return opinion_class_logits

class Step_2_forward0(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward0, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 用于将解码器输出的特征映射到具体的情感极性标签的数字 ID 上
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.reverse_decoder = Pointer_Block(args, self.bert_config, mask_for_encoder=False)
        self.dense = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)

    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor,opinion_prob,reverse_embedding):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output

        # reverse_embedding,_ = self.reverse_decoder(reverse_embedding,
        #                                            aspect_span_mask,
        #                                            spans_aspect_tensor)
        # reverse_embedding, _ = self.reverse_decoder(aspect_spans_embedding,
        #                                             aspect_span_mask,
        #                                             reverse_embedding)

        final_rep = self.dense(torch.cat((reverse_embedding,aspect_spans_embedding),dim=-1))
        opinion_class_logits = self.opinion_docoder2class(final_rep)
        return opinion_class_logits

class Step_2_reverse1(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse1, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 将解码器输出的特征映射到具体的情感极性标签上
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.dense = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)
        self.forward_decoder = Pointer_Block(args, self.bert_config, mask_for_encoder=False)
        self.output = Output(bert_config)
        self.intermediate = Intermediate(bert_config)
    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor,aspect_prob,forward_embedding):
        '''opinion---> aspect 方向'''
        # 将reverse_spans_embedding输入到解码器循环解码
        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 aspect 的预测结果，
        # 使用 aspect_docoder2class 层将其映射到具体的情感极性标签上，得到 aspect_class_logits。
        # input_rep = torch.cat((reverse_spans_embedding,all_left_tensor,all_right_tensor),dim=0)
        # aspect_prob = aspect_prob.unsqueeze(-1)
        # prob_embedding = self.dense(torch.mul(aspect_prob, reverse_spans_embedding))
        # final_embedding = torch.cat((reverse_spans_embedding,forward_embedding),dim=-1)
        # final_embedding = self.dense(final_embedding)

        forward_embedding, _ = self.forward_decoder(forward_embedding,
                                                    reverse_span_mask,
                                                    all_reverse_opinion_tensor)
        forward_embedding = self.intermediate(forward_embedding)
        final_rep = self.output(forward_embedding,reverse_spans_embedding)
        aspect_class_logits = self.aspect_docoder2class(final_rep)
        return aspect_class_logits

class Step_2_reverse(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 将解码器输出的特征映射到具体的情感极性标签上
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor,aspect_prob,forward_embedding):
        '''opinion---> aspect 方向'''
        # 将reverse_spans_embedding输入到解码器循环解码
        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        aspect_class_logits = self.aspect_docoder2class(reverse_spans_embedding)
        return aspect_class_logits


class Step_2_reverse0(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse0, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # 循环构建解码器
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 将解码器输出的特征映射到具体的情感极性标签上
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.dense = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)
        self.forward_decoder = Pointer_Block(args, self.bert_config, mask_for_encoder=False)
    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor,aspect_prob,forward_embedding):
        '''opinion---> aspect 方向'''
        # 将reverse_spans_embedding输入到解码器循环解码
        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output

        # forward_embedding, _ = self.forward_decoder(forward_embedding,
        #                                             reverse_span_mask,
        #                                             all_reverse_opinion_tensor)
        forward_embedding, _ = self.forward_decoder(reverse_spans_embedding,
                                                    reverse_span_mask,
                                                    forward_embedding)
        final_rep = self.dense(torch.cat((forward_embedding, reverse_spans_embedding), dim=-1))
        aspect_class_logits = self.aspect_docoder2class(final_rep)
        return aspect_class_logits
class Step_3_categories(torch.nn.Module):
    def __init__(self, args):
        super(Step_3_categories, self).__init__()
        self.args = args
        #self.fc = nn.Linear(768*2, len(categories))
        #self.dropout = nn.Dropout(0.5)
        categories_class = 13 if args.dataset == "restaurant" else 121
        self.category_classifier = nn.Sequential(
            nn.Linear(args.bert_feature_dim * 3, args.bert_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim * 2, args.bert_feature_dim),
            nn.ReLU(),
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim, categories_class)
        )
    def forward(self,spans_embedding,bert_spans_tensor,pairs,span_masks,category_labels=None):
        if category_labels == None:
            category_logits = self.category_classifier(pairs)
            return category_logits,[]
        else:
            input_rep = []
            all_left_list = []
            all_right_list = []
            for i,pair in enumerate(pairs):
                real_pair = self.cal_real(pair,bert_spans_tensor[i])
                last_index = torch.sum(span_masks[i]).item()-1
                for pair in real_pair:
                    aspect_index = int(pair[0])
                    opinion_index = int(pair[1])
                    # if aspect_index == 0 and opinion_index != 0:
                    #     left = torch.mean(spans_embedding[i,1:opinion_index,:],dim=0) if opinion_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i,opinion_index:last_index,:],dim=0) if opinion_index != last_index else spans_embedding[i][last_index]
                    # elif aspect_index != 0 and opinion_index == 0:
                    #     left = torch.mean(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i, aspect_index:last_index, :], dim=0) if aspect_index != last_index else spans_embedding[i][last_index]
                    # elif aspect_index == 0 and opinion_index == 0:
                    #     left = torch.mean(spans_embedding[i, 1:last_index, :], dim=0)
                    #     right,_ = torch.max(spans_embedding[i, 1:last_index, :], dim=0)
                    # else:
                    #     left = torch.mean(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i, aspect_index:last_index, :], dim=0) if aspect_index != last_index else spans_embedding[i][last_index]
                    if aspect_index == 0 and opinion_index != 0:
                        left, _ = torch.max(spans_embedding[i, 1:opinion_index, :], dim=0) if opinion_index != 1 else (spans_embedding[i][0], 0)
                        right, _ = torch.max(spans_embedding[i, opinion_index:last_index, :],dim=0) if opinion_index != last_index else (spans_embedding[i][last_index], 0)
                    elif aspect_index != 0 and opinion_index == 0:
                        left, _ = torch.max(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else (spans_embedding[i][0], 0)
                        right, _ = torch.max(spans_embedding[i, aspect_index:last_index, :],dim=0) if aspect_index != last_index else (spans_embedding[i][last_index], 0)
                    elif aspect_index == 0 and opinion_index == 0:
                        left = torch.mean(spans_embedding[i, 1:last_index, :], dim=0)
                        right, _ = torch.max(spans_embedding[i, 1:last_index, :], dim=0)
                    else:
                        left, _ = torch.max(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else (spans_embedding[i][0], 0)
                        right, _ = torch.max(spans_embedding[i, aspect_index:last_index, :],dim=0) if aspect_index != last_index else (spans_embedding[i][last_index], 0)

                    aspect_rep = spans_embedding[i][aspect_index]
                    opinion_index = opinion_index if opinion_index!=0 else last_index
                    opinion_rep = spans_embedding[i][opinion_index]
                    all_left_list.append(left)
                    all_right_list.append(right)
                    final_rep = torch.unsqueeze(torch.cat((aspect_rep+opinion_rep,left,right),dim=0),0)
                    # final_rep = torch.unsqueeze(aspect_rep+opinion_rep,0)
                    if input_rep == []:
                        input_rep = final_rep
                    else:
                        # input_rep= torch.cat((input_rep,final_rep),dim=0)
                        input_rep = torch.cat((input_rep, final_rep), dim=0)
            category_logits = self.category_classifier(input_rep)
                # category_logits = self.dropout(category_logits)
            category_label = torch.cat(category_labels)
            all_left_tensor = torch.stack(all_left_list)
            all_right_tensor = torch.stack(all_right_list)
            return category_logits,category_label
    def cal_real(self,ao_pair,spans):
        real_pair = []
        ao_pair = torch.tensor(ao_pair)
        for span in ao_pair:
            span = span.clone().detach().to(self.args.device)
            aspect_indices = torch.nonzero(torch.all(torch.eq(spans, span[0]), dim=1))
            # 如果有匹配的索引，返回第一个匹配的索引；如果没有匹配的索引，返回0
            aspect_index = aspect_indices[0][0].item() if aspect_indices.numel() > 0 else 0
            opinion_indices = torch.nonzero(torch.all(torch.eq(spans, span[1]), dim=1))
            # 如果有匹配的索引，返回第一个匹配的索引；如果没有匹配的索引，返回0
            opinion_index = opinion_indices[0][0].item() if opinion_indices.numel() > 0 else 0
            real_pair.append([aspect_index,opinion_index])
        return real_pair



def Loss(gold_aspect_label, pred_aspect_label, gold_opinion_label, pred_opinion_label, spans_mask_tensor, opinion_span_mask_tensor,
         reverse_gold_opinion_label, reverse_pred_opinion_label, reverse_gold_aspect_label, reverse_pred_aspect_label,
         cnn_spans_mask_tensor, reverse_aspect_span_mask_tensor, spans_embedding, related_spans_tensor,gold_category_label,
         pred_category_label, exist_aspect,exist_opinion,imp_aspect_exist,imp_opinion_exist,sentence_length,
        opinion_soft,forward_softmax,aspect_soft,reverse_softmax,
         args):
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    if cnn_spans_mask_tensor is not None:
        spans_mask_tensor = cnn_spans_mask_tensor
    # pos_weight = torch.tensor([args.binary_weight]).to(args.device)
    # Loss正向
    aspect_spans_mask_tensor = spans_mask_tensor.view(-1) == 1
    pred_aspect_label_logits = pred_aspect_label.view(-1, pred_aspect_label.shape[-1])
    gold_aspect_effective_label = torch.where(aspect_spans_mask_tensor, gold_aspect_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(gold_aspect_label))
    aspect_loss = loss_function(pred_aspect_label_logits, gold_aspect_effective_label)


    # pred_category_label_logits = pred_category_label.view(-1,pred_category_label.shape[-1])
    # gold_category_effective_label = torch.where(aspect_spans_mask_tensor, gold_category_label.view(-1),
    #                                           torch.tensor(loss_function.ignore_index).type_as(gold_category_label))
    # category_loss = 0
    # if pred_category_label != [] and gold_category_label != []:
    category_loss = loss_function(pred_category_label,gold_category_label)

    # pred_category_label_logits = pred_category_label.view(-1, pred_category_label.shape[-1])
    # gold_category_effective_label = torch.where(aspect_spans_mask_tensor, gold_category_label.view(-1),
    #                                           torch.tensor(loss_function.ignore_index).type_as(gold_category_label))
    # category_loss = loss_function(pred_category_label_logits, gold_category_effective_label)


    opinion_span_mask_tensor = opinion_span_mask_tensor.view(-1) == 1
    pred_opinion_label_logits = pred_opinion_label.view(-1, pred_opinion_label.shape[-1])
    gold_opinion_effective_label = torch.where(opinion_span_mask_tensor, gold_opinion_label.view(-1),
                                               torch.tensor(loss_function.ignore_index).type_as(gold_opinion_label))
    opinion_loss = loss_function(pred_opinion_label_logits, gold_opinion_effective_label)
    as_2_op_loss = aspect_loss + opinion_loss + category_loss

    # Loss反向
    reverse_opinion_span_mask_tensor = spans_mask_tensor.view(-1) == 1
    reverse_pred_opinion_label_logits = reverse_pred_opinion_label.view(-1, reverse_pred_opinion_label.shape[-1])
    reverse_gold_opinion_effective_label = torch.where(reverse_opinion_span_mask_tensor, reverse_gold_opinion_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(reverse_gold_opinion_label))
    reverse_opinion_loss = loss_function(reverse_pred_opinion_label_logits, reverse_gold_opinion_effective_label)

    reverse_aspect_span_mask_tensor = reverse_aspect_span_mask_tensor.view(-1) == 1
    reverse_pred_aspect_label_logits = reverse_pred_aspect_label.view(-1, reverse_pred_aspect_label.shape[-1])
    reverse_gold_aspect_effective_label = torch.where(reverse_aspect_span_mask_tensor, reverse_gold_aspect_label.view(-1),
                                               torch.tensor(loss_function.ignore_index).type_as(reverse_gold_aspect_label))
    reverse_aspect_loss = loss_function(reverse_pred_aspect_label_logits, reverse_gold_aspect_effective_label)
    op_2_as_loss = reverse_opinion_loss + reverse_aspect_loss
    imp_aspect_loss = loss_function(imp_aspect_exist, exist_aspect.view(-1))
    imp_opinion_loss = loss_function(imp_opinion_exist, exist_opinion.view(-1))

    # imp_asp_label_tensor = F.one_hot(imp_asp_label_tensor, num_classes=2).float()
    # imp_asp_loss = imp_criterion(aspect_imp_logits, imp_asp_label_tensor)
    # imp_opi_label_tensor = F.one_hot(imp_opi_label_tensor, num_classes=2).float()
    # imp_opi_loss = imp_criterion(opinion_imp_logits, imp_opi_label_tensor)
    # imp_opi_loss = loss_function(aspect_imp_logits,imp_asp_label_tensor.view(-1))
    # F.one_hot(imp_asp_label_tensor, num_classes=2)

    # aspect_polarity_loss = loss_function(aspect_imp_polarity_logits,aspect_polarity_label_tensor.view(-1))
    # opinion_polarity_loss = loss_function(opinion_imp_polarity_logits,opinion_polarity_label_tensor.view(-1))
    # prob_loss = L1(opinion_soft,forward_softmax)+L1(aspect_soft,reverse_softmax)
    if args.kl_loss:
        kl_loss = shape_span_embedding(args, spans_embedding, spans_embedding, related_spans_tensor, spans_mask_tensor,sentence_length)
        # loss = as_2_op_loss + op_2_as_loss + kl_loss
        loss = as_2_op_loss + op_2_as_loss + args.kl_loss_weight * kl_loss + imp_aspect_loss + imp_opinion_loss
    else:
        loss = as_2_op_loss + op_2_as_loss + imp_aspect_loss + imp_opinion_loss
        kl_loss = 0


    return loss,  args.kl_loss_weight * kl_loss

def shape_span_embedding(args, p, q, pad_mask, span_mask,sentence_length):
    kl_loss = 0
    input_size = p.size()
    assert input_size == q.size()
    for i in range(input_size[0]):
        span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
        if len(span_mask_index.shape) == 0:
            return 0
        flag = True
        lucky_squence = random.choice(span_mask_index)
        if lucky_squence ==0 and lucky_squence==len(span_mask_index)-1:
            continue
        P = p[i, lucky_squence, :]
        mask_index = torch.nonzero(pad_mask[i, lucky_squence, :])
        q_tensor = None
        for idx in mask_index:
            if idx == lucky_squence:
                continue
            if q_tensor is None:
                q_tensor = p[i, idx]
            else:
                q_tensor = torch.cat((q_tensor, p[i, idx]), dim=0)
        if q_tensor is None:
            continue
        expan_P = P.expand_as(q_tensor)
        kl_loss += compute_kl_loss(args, expan_P, q_tensor)
    return kl_loss

def shape_span_embedding_V2(args, p, q, pad_mask, span_mask,sentence_length):
    kl_loss = 0
    input_size = p.size()
    assert input_size == q.size()
    for i in range(input_size[0]):
        span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
        for j in range(sentence_length[i][1]):
            if j == 0 or j == sentence_length[i][1]-1:
                continue
            P = p[i, j, :]
            mask_index = torch.nonzero(pad_mask[i, j, :])
            q_tensor = None
            for idx in mask_index:
                if idx == j:
                    continue
                if q_tensor is None:
                    q_tensor = p[i, idx]
                else:
                    q_tensor = torch.cat((q_tensor, p[i, idx]), dim=0)
            if q_tensor is None:
                continue
            expan_P = P.expand_as(q_tensor)
            kl_loss += compute_kl_loss(args, expan_P, q_tensor)
    return kl_loss

def shape_span_embedding_V3(args, p, q, pad_mask, span_mask,sentence_length):
    kl_loss = 0
    input_size = p.size()
    assert input_size == q.size()
    for i in range(input_size[0]):
        span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
        time = 2 if sentence_length[i][1]<5 else 5
        for j in range(time):
            if j == 0:
                continue
            lucky_squence = random.choice(span_mask_index)
            P = p[i, lucky_squence, :]
            mask_index = torch.nonzero(pad_mask[i, lucky_squence, :])
            q_tensor = None
            for idx in mask_index:
                if idx == j:
                    continue
                if q_tensor is None:
                    q_tensor = p[i, idx]
                else:
                    q_tensor = torch.cat((q_tensor, p[i, idx]), dim=0)
            if q_tensor is None:
                continue
            expan_P = P.expand_as(q_tensor)
            kl_loss += compute_kl_loss(args, expan_P, q_tensor)
    return kl_loss

def compute_kl_loss(args, p, q, pad_mask=None):
    if args.kl_loss_mode == "KLLoss":
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill(pad_mask, 0.)
            q_loss.masked_fill(pad_mask, 0.)
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        if (p_loss + q_loss)==0:
            return 0
        total_loss = math.log(1+5/((p_loss + q_loss) / 2))
    elif args.kl_loss_mode == "JSLoss":
        m = (p+q)/2
        m_loss = 0.5 * F.kl_div(F.log_softmax(p, dim=-1), F.softmax(m, dim=-1), reduction="none") + 0.5 * F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(m, dim=-1), reduction="none")
        if pad_mask is not None:
            m_loss.masked_fill(pad_mask, 0.)
        m_loss = m_loss.sum()
        # test = -math.log(2*m_loss)-math.log(-2*m_loss+2)
        total_loss = 10*(math.log(1+5/m_loss))
    elif args.kl_loss_mode == "EMLoss":
        test = torch.square(p-q)
        em_loss = torch.sqrt(torch.sum(torch.square(p - q)))
        total_loss = math.log(1+5/(em_loss))
    elif args.kl_loss_mode == "CSLoss":
        test = torch.cosine_similarity(p, q, dim=1)
        cs_loss = torch.sum(torch.cosine_similarity(p, q, dim=1))
        total_loss = math.log(1 + 5 / (cs_loss))
    else:
        total_loss = 0
        print('损失种类错误')
    return  total_loss




if __name__ == '__main__':
    tensor1 = torch.zeros((3,3))
    tensor2 = torch.nonzero(tensor1, as_tuple=False)
    tensor1 = tensor1.type_as(tensor2)
    print('666')
