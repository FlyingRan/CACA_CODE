import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
# from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .Attention import Attention, Intermediate, Output, Dim_Four_Attention, masked_softmax,implict_ATT,SelfAttention
from .data_BIO_loader import sentiment2id, validity2id
from allennlp.nn.util import batched_index_select, batched_span_select
from .kan import *
import random
import math


def modify_range(tensor, left_shift, right_shift, max_value):
    # 往左边移动第一个值
    shifted_left_0 = max(0, tensor[0] - left_shift)

    # 往右边移动第二个值
    shifted_right_1 = min(max_value, tensor[1] + right_shift)

    return torch.tensor([shifted_left_0, shifted_right_1])
def stage_2_features_generation1(bert_feature, attention_mask, spans, span_mask, spans_embedding, spans_embedding1,spans_aspect_tensor,
                                is_aspect,imp_rep,spans_opinion_tensor=None):
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
        last = torch.sum(span_mask[int(batch_num)]).item() - 1
        last_att = torch.sum(attention_mask[int(batch_num)]).item()-1
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        if (test[1] == test[2] == -1) or (test[1] == test[2] == 0) or (test[1] == test[2] == last_att):
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
            help = torch.mean(spans_embedding1[batch_num, 0:last, :], dim=0).unsqueeze(0)
            if is_aspect:
                # aspect_span_embedding_unspilt = spans_embedding[batch_num, torch.tensor([0]), :].unsqueeze(0)
                # aspect_span_embedding_unspilt = torch.mean(torch.stack([spans_embedding[batch_num, torch.tensor([0]), :],imp_rep[batch_num].unsqueeze(0)]),dim=0).unsqueeze(0)
                # aspect_span_embedding_unspilt = (spans_embedding[batch_num, torch.tensor([0]), :]+imp_rep[batch_num]).unsqueeze(0)
                aspect_span_embedding_unspilt = imp_rep[batch_num].unsqueeze(0).unsqueeze(0)
            else:
                # aspect_span_embedding_unspilt = spans_embedding[batch_num,  torch.tensor([last]), :].unsqueeze(0)
                # aspect_span_embedding_unspilt = torch.mean(torch.stack([spans_embedding[batch_num, torch.tensor([last]), :],imp_rep[batch_num].unsqueeze(0)]), dim=0).unsqueeze(0)
                # aspect_span_embedding_unspilt = (spans_embedding[batch_num, torch.tensor([last]), :]+imp_rep[batch_num]).unsqueeze(0)
                aspect_span_embedding_unspilt = (imp_rep[batch_num].unsqueeze(0)).unsqueeze(0)
            flag = 0

        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        spans_embedding_unspilt1 = spans_embedding1[batch_num,:,:].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)

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
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            if len(all_span_aspect_tensor.shape)!=len(aspect_span_embedding_unspilt.shape):
                print(1)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)

            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            # all_left_tensor = torch.cat((all_left_tensor, left), dim=0)
            # all_right_tensor = torch.cat((all_right_tensor, right), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_spans_embedding1 = torch.cat((all_spans_embedding1, spans_embedding_unspilt1), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
           all_spans_embedding, all_spans_embedding1,all_span_mask,all_left_tensor,all_right_tensor


class RelativePositionEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_length):
        super(RelativePositionEncoding, self).__init__()
        self.relative_positions = nn.Parameter(torch.randn(max_seq_length, max_seq_length, hidden_size))

    def forward(self, input, mask=None):
        batch_size, seq_length, hidden_size = input.size()
        relative_positions = self.relative_positions[:seq_length, :seq_length, :]  # Select relevant positions

        # Expand dimensions for batch and heads
        input = input.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length, hidden_size)
        relative_positions = relative_positions.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length, hidden_size)

        # Compute attention scores with relative positions
        attention_scores = torch.sum(input * relative_positions, dim=-1)

        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of input elements with attention weights
        output = torch.sum(attention_weights.unsqueeze(-1) * input, dim=-2).squeeze(1).squeeze(1)

        return output

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, args,max_len):
        super(AbsolutePositionalEncoding, self).__init__()
        self.encoding = self.create_encoding(max_len, args.bert_feature_dim)
        self.args = args
    def create_encoding(self, max_len, embedding_dim):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        encoding = torch.zeros(max_len, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        return encoding

    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.size()
        y = self.encoding[:, :seq_len].detach().to(self.args.device)
        x = x + y

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

    def forward(self, start,end):
        # H 形状为 (batch, num_nodes, input_dim)
        # batch_size, num_nodes, input_dim = H.size()

        # 计算注意力权重矩阵 e_ij
        # H_transformed = torch.matmul(H, self.W1)  # (batch, num_nodes, input_dim)
        # attention_scores = torch.matmul(H_transformed, H_transformed.transpose(2, 3))  # (batch, num_nodes, num_nodes)
        span_start_transformed = torch.matmul(start, self.W1)  # (batch, num_spans, input_dim)
        attention_scores = torch.matmul(span_start_transformed,
                                        end.transpose(1, 2))  # (batch, num_spans, num_spans)

        # 归一化权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights


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



        self.args = args
        self.bert_config = bert_config
        self.dropout_output = torch.nn.Dropout(args.drop_out)
        self.forward_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)

        # self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.reverse_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])

        # self.aspect_imp_decoders = nn.ModuleList(
        #     [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        #
        # self.opinion_imp_decoders = nn.ModuleList(
        #     [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])

        self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)
        # self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        # self.categories_classification_opinion = nn.Linear(args.bert_feature_dim, len(categories))
        # self.ATT_attentions = nn.ModuleList(
        #     [Dim_Four_Block(args, self.bert_config) for _ in range(max(1, args.ATT_SPAN_block_num - 1))])
        # self.multhead_asp = Pointer_Block(args, self.bert_config, mask_for_encoder=False)

        # self.multhead_opi = Pointer_Block(args, self.bert_config, mask_for_encoder=False)
        self.atten_imp = nn.MultiheadAttention(args.bert_feature_dim,4)
        self.imp_asp_mlp = nn.Linear(args.bert_feature_dim, args.bert_feature_dim)
        self.imp_opi_mlp = nn.Linear(args.bert_feature_dim, args.bert_feature_dim)
        self.imp_asp_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim, 2)
        )
        self.imp_opi_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.bert_feature_dim, 2)
        )
        # self.textCNN_A = TextCNN(args.bert_feature_dim,2,[3,4,5],3)
        # self.textCNN_O = TextCNN(args.bert_feature_dim,2,[3,4,5],3)
        # 创建 intermediate 层，用于对 attention_output 进行非线性变换，提高特征表达的能力
        self.intermediate = Intermediate(bert_config)

        # 创建 output 层，用于将经过 intermediate 层处理之后的特征映射到实际输出空间中
        # self.output_a = Output(bert_config)
        # self.output_o = Output(bert_config)
        # self.compess_projection = nn.Sequential(nn.Linear(args.bert_feature_dim, 1), nn.ReLU(),
        #                                             nn.Dropout(args.drop_out))

        # self.opinion_lstm = HierarchicalLSTMWithAttention(args.bert_feature_dim,128,2)
        # self.aspect_lstm = HierarchicalLSTMWithAttention(args.bert_feature_dim, 12o8, 2)

        # self._densenet = nn.Sequential(nn.Linear(args.bert_feature_dim*4, args.bert_feature_dim),
        #                                nn.Tanh())
        # self._densenet = nn.Linear(args.bert_feature_dim*4, args.bert_feature_dim)
        self._densenet = nn.Linear(args.bert_feature_dim*4, args.bert_feature_dim)
        self.attention_layer = AttentionLayer(args.bert_feature_dim)
        # self.imp_classifier_asp = imp_classifier(args.bert_feature_dim,128,[3, 4, 5],2,0.5)
        # self.imp_classifier_opi = imp_classifier(args.bert_feature_dim, 128, [3, 4, 5], 2, 0.5)

        # self.imp_classifier_asp = AttentionCNN(args.bert_feature_dim,64,3,64)
        # self.imp_classifier_opi = AttentionCNN(args.bert_feature_dim,64,3,64)
        # 一致性
        # self.imp_asp_classifier1 = nn.Sequential(
        #     nn.Dropout(args.drop_out),
        #     nn.Linear(args.bert_feature_dim, 2)
        # )
        # self.imp_opi_classifier1 = nn.Sequential(
        #     nn.Dropout(args.drop_out),
        #     nn.Linear(args.bert_feature_dim, 2)
        # )

        #self.bert_blend_cnn_asp = Bert_Blend_CNN(args)
        #self.bert_blend_cnn_opi = Bert_Blend_CNN(args)

        # position encodeing
        # self.position_encoder = RelativePositionEncoding(args.bert_feature_dim,428)
        #self.position_encoder = AbsolutePositionalEncoding(args,1000)

    def forward(self,bert_output, attention_mask, spans, span_mask, related_spans_tensor,sentence_length):
        # 调用 span_generator 方法生成 spans_embedding 和 features_mask_tensor 两个变量。
        # 其中 spans_embedding 表示每个 span 的嵌入表示，features_mask_tensor 表示每个 span 是否被标记为有效的情感单元。
        # test = self.opinion_lstm(input_bert_features)
        bert_rsp = bert_output.last_hidden_state
        spans_embedding, features_mask_tensor = self.span_generator(bert_output.last_hidden_state, attention_mask, spans,
                                                                    span_mask, related_spans_tensor, sentence_length)
        #spans_embedding = spans_embedding+self.position_encoder(spans_embedding,span_mask)
        # spans_embedding = spans_embedding
        # imp_aspect_exist, asp_embedding = self.bert_blend_cnn_asp(bert_output)
        # imp_opinion_exist, opi_embedding = self.bert_blend_cnn_opi(bert_output)
        # spans_embedding[:, 0, :] =  asp_embedding + spans_embedding[:, 0, :]
        batch_size = spans_embedding.shape[0]
        # 循环遍历每个批次
        # for i in range(batch_size):
        #     # 获取当前批次的索引
        #     index = torch.sum(span_mask[i], dim=-1) - 1
        #     # 赋值操作
        #     spans_embedding[i, index] = opi_embedding[i] + spans_embedding[i, index]

        # # input_vector1 = output1[:,0,:].unsqueeze(1)
        # # input_vector2 = output2[range(span_embedding_4.shape[0]), torch.sum(span_mask, dim=-1) - 1].unsqueeze(1)
        # # input_vector2 = input_vector2.view(span_embedding_4.shape[0], -1)
        # # input_vector1 = input_vector1.view(span_embedding_3.shape[0], -1)
        span_embedding_1 = torch.clone(spans_embedding)
        span_embedding_2 = torch.clone(spans_embedding)
        span_embedding_3 = torch.clone(spans_embedding)
        key_padding_mask = torch.logical_not(span_mask)
        embedding = torch.mean(self.atten_imp(span_embedding_3.permute(1,0,2),
            span_embedding_3.permute(1, 0, 2),span_embedding_3.permute(1,0,2),key_padding_mask=key_padding_mask)[0].permute(1,0,2),dim=1)
        embedding_1 = self.imp_asp_mlp(embedding)
        embedding_2 = self.imp_opi_mlp(embedding)
        imp_aspect_exist = self.imp_asp_classifier(embedding_1)
        imp_opinion_exist = self.imp_opi_classifier(embedding_2)

        # global_rep1,imp_aspect_exist = self.textCNN_A(bert_rsp)
        # global_rep2,imp_opinion_exist = self.textCNN_O(bert_rsp)

        for i in range(batch_size):
            # 获取当前批次的索引
            index = torch.sum(span_mask[i], dim=-1) - 1
            # 赋值操作
            span_embedding_1[i, 0] = embedding_1[i].squeeze()
            span_embedding_1[i, index] = embedding_2[i].squeeze()
            span_embedding_2[i, index] = embedding_2[i].squeeze()
            span_embedding_2[i, 0] = embedding_1[i].squeeze()


        # for i in range(batch_size):
        #     # 获取当前批次的索引
        #     index = torch.sum(span_mask[i], dim=-1) - 1
        #     # 赋值操作
        #     # span_embedding_1[i, 0] = input_vector2[i].squeeze()
        #     span_embedding_1[i, index] = input_vector2[i].squeeze()
        #     # span_embedding_2[i, index] = output2[i].squeeze()
        #     span_embedding_2[i, 0] = input_vector1[i].squeeze()

        '''
        intermediate_a = self.intermediate(pooler_output.clone())
        intermediate_o = self.intermediate(pooler_output.clone())
        global_rep1 = self.output_a(intermediate_a,pooler_output.clone())
        global_rep2 = self.output_o(intermediate_o,pooler_output.clone())
        imp_aspect_exist = self.imp_asp_classifier(global_rep1)
        imp_opinion_exist = self.imp_opi_classifier(global_rep2)
        for i in range(batch_size):
            # 获取当前批次的索引
            index = torch.sum(span_mask[i], dim=-1) - 1
            # 赋值操作
            span_embedding_1[i, 0] = global_rep1[i].squeeze()
            span_embedding_1[i, index] = global_rep2[i].squeeze()
            span_embedding_2[i, index] = global_rep2[i].squeeze()
            span_embedding_2[i, 0] = global_rep1[i].squeeze()
        '''

        for forward_1_decoder in self.forward_1_decoders:
            forward_layer_output, forward_intermediate_output = forward_1_decoder(span_embedding_1)
            span_embedding_1 = forward_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 aspect 的预测结果，
        # 使用 sentiment_classification_aspect 层将其映射到具体的情感极性标签上，得到 class_logits_aspect。
        class_logits_aspect = self.sentiment_classification_aspect(span_embedding_1)
        # class_logits_category = self.categories_classification_opinion(span_embedding_1)
        # 复制 spans_embedding 得到 span_embedding_1 和 span_embedding_2 两个变量，
        # 然后分别将它们输入到 forward_1_decoders 和 reverse_1_decoders 中进行解码。

        for reverse_1_decoder in self.reverse_1_decoders:
            reverse_layer_output, reverse_intermediate_output = reverse_1_decoder(span_embedding_2)
            span_embedding_2 = reverse_layer_output
        # 同样用 sentiment_classification_opinion 层将其映射到具体的情感极性标签上，得到 class_logits_opinion。
        class_logits_opinion = self.sentiment_classification_opinion(span_embedding_2)

        '''
        span_embedding_3 = torch.clone(span_embedding_1)
        span_embedding_4 = torch.clone(span_embedding_2)
        output1, _ = self.multhead_asp(span_embedding_3, span_mask, pooler_output.unsqueeze(1))
        output2, _ = self.multhead_opi(span_embedding_4, span_mask, pooler_output.unsqueeze(1))
        # query1 = span_embedding_3[:, 0, :]
        # query2 = span_embedding_4[range(span_embedding_4.shape[0]), torch.sum(span_mask, dim=-1) - 1]

        input_vector1 = torch.mean(output1, dim=1)
        input_vector2 = torch.mean(output2, dim=1)
        # output1 = output1.squeeze()
        # output2 = output2.squeeze()
        imp_aspect_exist = self.imp_asp_classifier(input_vector1)
        imp_opinion_exist = self.imp_opi_classifier(input_vector2)
        '''

        '''
        span_embedding_3 = torch.clone(span_embedding_1)
        span_embedding_4 = torch.clone(span_embedding_2)
        output1, _ = self.multhead_asp(span_embedding_3, span_mask, pooler_output.unsqueeze(1))
        output2, _ = self.multhead_opi(span_embedding_4, span_mask, pooler_output.unsqueeze(1))
        # query1 = span_embedding_3[:, 0, :]
        # query2 = span_embedding_4[range(span_embedding_4.shape[0]), torch.sum(span_mask, dim=-1) - 1]

        input_vector1 = torch.mean(output1,dim=1)
        input_vector2 = torch.mean(output2,dim=1)
        # output1 = output1.squeeze()
        # output2 = output2.squeeze()
        
        imp_aspect_exist = self.imp_asp_classifier(input_vector1)
        imp_opinion_exist = self.imp_opi_classifier(input_vector2)
        '''

        return class_logits_aspect, class_logits_opinion, spans_embedding, span_embedding_1, span_embedding_2, \
               features_mask_tensor,imp_aspect_exist,imp_opinion_exist,embedding_1,embedding_2


    def span_generator(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor,
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
                spans_width_start_end[i, last_index][0] = sum(attention_mask[i]) - 1
                spans_width_start_end[i, last_index][1] = sum(attention_mask[i]) - 1
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
            spans_num = spans.shape[1]  # 有多少个span 包括了前后的【0，0，0】
            spans_width_start_end = spans[:, :, 0:2].view(spans.size(0), spans_num, -1)  # 取所有span的前两位
            spans_width_start_end[0, -1] = spans_width_start_end[0, -2].clone()
            last_one_index = span_mask.flip(dims=(1,)).argmax(dim=1)
            last_one_index = span_mask.size(1) - 1 - last_one_index
            for i in range(len(span_mask)):
                last_index = int(last_one_index[i])
                spans_width_start_end[i, last_index][0] = sum(attention_mask[i]) - 1
                spans_width_start_end[i, last_index][1] = sum(attention_mask[i]) - 1
                spans_start[i][last_index] = sum(attention_mask[i]) - 1
                spans_end[i][last_index] =  sum(attention_mask[i]) - 1

            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))
            spans_width_start_end_mean = spans_width_start_end_embedding.mean(dim=2, keepdim=True).squeeze(-2)

            spans_start_embedding = batched_index_select(bert_feature, spans_start)
            spans_end_embedding = batched_index_select(bert_feature, spans_end)
            spans_embedding = torch.cat((spans_start_embedding,spans_end_embedding,spans_end_embedding-spans_start_embedding,spans_width_start_end_mean),dim=-1)
            # spans_start_end_att = self.attention_layer(spans_start_embedding,spans_end_embedding)
            # spans_embedding = torch.cat((spans_start_embedding,spans_end_embedding,spans_width_start_end_mean,spans_start_end_att),dim=-1)
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
                                                       lstm_states = outputs,
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


class Step_2_forward0(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward0, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # self.cross_att = Attention(bert_config)
        # 循环构建解码器
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 用于将解码器输出的特征映射到具体的情感极性标签的数字 ID 上
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        # self.linear = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)

    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor,reverse_embedding):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        # masks = (1 - aspect_span_mask) * -1e9
        # # 根据掩码的维度信息确定 attention_masks 的形状
        # masks = masks[:, None, :,None]
        # reverse_spans_embedding = self.cross_att(reverse_embedding,
        #                                         lstm_states =reverse_embedding,
        #                                         encoder_hidden_states=aspect_spans_embedding,
        #                                         encoder_attention_mask=masks)[0]
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output
        # opinion_class_logits = self.opinion_docoder2class(self.linear(torch.cat((aspect_spans_embedding,reverse_spans_embedding),dim=-1)))
        opinion_class_logits = self.opinion_docoder2class(aspect_spans_embedding)


        return opinion_class_logits
class Step_2_forward(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.cross_att = Attention(bert_config)
        # 循环构建解码器
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 用于将解码器输出的特征映射到具体的情感极性标签的数字 ID 上
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.linear = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)
        # self.GCN = GCN(args.bert_feature_dim,args.bert_feature_dim*2,args.bert_feature_dim)
        # self.attention = nn.MultiheadAttention(args.bert_feature_dim, 4)
    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor,reverse_embedding):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        masks = (1 - aspect_span_mask) * -1e9
        # 根据掩码的维度信息确定 attention_masks 的形状
        masks = masks[:, None, :,None]
        aspect_spans_embedding1 = aspect_spans_embedding.clone()
        aspect_spans_embedding2 = aspect_spans_embedding.clone()
        reverse_embedding1 = reverse_embedding.clone()
        reverse_spans_embedding = self.cross_att(aspect_spans_embedding1,
                                                lstm_states =aspect_spans_embedding1,
                                                encoder_hidden_states=reverse_embedding1,
                                                encoder_attention_mask=masks)[0]

        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output
        # key_padding_mask = torch.logical_not(aspect_span_mask.bool())
        # attention_output,attention_weights = self.attention(aspect_spans_embedding2.permute(1,0,2),
        #                                                     aspect_spans_embedding2.permute(1,0,2),
        #                                                     aspect_spans_embedding2.permute(1,0,2),
        #                                                     key_padding_mask=key_padding_mask,
        #                                                     )
        # attention_weights = self.attention.attention.forward_self_attention.attention_probs
        # gcn_output = self.GCN(aspect_spans_embedding,attention_weights)
        opinion_class_logits = self.opinion_docoder2class(self.linear(torch.cat((aspect_spans_embedding,reverse_spans_embedding),dim=-1)))
        # opinion_class_logits = self.opinion_docoder2class(aspect_spans_embedding)
        return opinion_class_logits

class Step_2_reverse(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.cross_att = Attention(bert_config)
        # 循环构建解码器
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 将解码器输出的特征映射到具体的情感极性标签上
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.linear = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)
        # self.GCN = GCN(args.bert_feature_dim, args.bert_feature_dim * 3, args.bert_feature_dim)
        # self.attention = nn.MultiheadAttention(args.bert_feature_dim, 4)
    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor,forward_embedding):

        '''opinion---> aspect 方向'''
        reverse_spans_embedding1 = reverse_spans_embedding.clone()
        reverse_spans_embedding2 = reverse_spans_embedding.clone()
        forward_embedding1 = forward_embedding.clone()
        # 将reverse_spans_embedding输入到解码器循环解码
        masks = (1 - reverse_span_mask) * -1e9
        # 根据掩码的维度信息确定 attention_masks 的形状
        masks = masks[:, None,: ,None]
        forward_spans_embedding = self.cross_att(reverse_spans_embedding1,
        lstm_states=forward_embedding,
        encoder_hidden_states=forward_embedding1,
        encoder_attention_mask=masks)[0]


        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        # key_padding_mask = torch.logical_not(reverse_span_mask.bool())
        # attention_output, attention_weights = self.attention(reverse_spans_embedding2.permute(1, 0, 2),
        #                                                      reverse_spans_embedding2.permute(1, 0, 2),
        #                                                      reverse_spans_embedding2.permute(1, 0, 2),
        #                                                      key_padding_mask = key_padding_mask)
        # attention_weights = self.attention.attention.forward_self_attention.attention_probs
        # gcn_output = self.GCN(reverse_spans_embedding, attention_weights)
        aspect_class_logits = self.aspect_docoder2class(self.linear(torch.cat((reverse_spans_embedding,forward_spans_embedding),dim=-1)))
        # aspect_class_logits = self.aspect_docoder2class(reverse_spans_embedding)
        return aspect_class_logits

class Step_2_reverse0(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse0, self).__init__()
        self.args = args
        self.bert_config = bert_config
        # self.cross_att = Attention(bert_config)
        # 循环构建解码器
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        # 将解码器输出的特征映射到具体的情感极性标签上
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        # self.linear = nn.Linear(args.bert_feature_dim*2,args.bert_feature_dim)

    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor,forward_embedding):

        '''opinion---> aspect 方向'''
        # 将reverse_spans_embedding输入到解码器循环解码
        # masks = (1 - reverse_span_mask) * -1e9
        # # 根据掩码的维度信息确定 attention_masks 的形状
        # masks = masks[:, None,: ,None]
        # forward_spans_embedding = self.cross_att(forward_embedding,
        # lstm_states=forward_embedding,
        # encoder_hidden_states=reverse_spans_embedding.clone(),
        # encoder_attention_mask=masks)[0]


        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        # aspect_class_logits = self.aspect_docoder2class(self.linear(torch.cat((reverse_spans_embedding,forward_spans_embedding),dim=-1)))
        aspect_class_logits = self.aspect_docoder2class(reverse_spans_embedding)
        return aspect_class_logits

class Step_3_categories(torch.nn.Module):
    def __init__(self, args):
        super(Step_3_categories, self).__init__()
        self.args = args
        #self.fc = nn.Linear(768*2, len(categories))
        #self.dropout = nn.Dropout(0.5)
        if args.dataset == "restaurant":
            categories_class = 13
        elif args.dataset == "laptop":
            categories_class = 121
        elif args.dataset == "phone":
            categories_class = 88
        else:
            categories_class = 2
        # categories_class = 13 if args.dataset == "restaurant" else 121
        # self.category_classifier = nn.Sequential(
        #     nn.Linear(args.bert_feature_dim * 3, args.bert_feature_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(args.drop_out),
        #     nn.Linear(args.bert_feature_dim * 2, args.bert_feature_dim),
        #     nn.ReLU(),
        #     nn.Dropout(args.drop_out),
        #     nn.Linear(args.bert_feature_dim, categories_class)
        # )
        # width = [2, 5, 3], grid = 5, k = 3
        # self.kan = KAN([args.bert_feature_dim*2, args.bert_feature_dim, categories_class])
        self.category_classifier = nn.Sequential(
            nn.Linear(args.bert_feature_dim * 2, args.bert_feature_dim),
            nn.ReLU(),
            nn.Dropout(args.drop_out),
            nn.LayerNorm(args.bert_feature_dim),
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
                    #     left = torch.mean(spans_embedding[i, 1:opinion_index, :], dim=0) if opinion_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i, opinion_index:last_index, :],dim=0) if opinion_index != last_index else spans_embedding[i][last_index]
                    # elif aspect_index != 0 and opinion_index == 0:
                    #     left = torch.mean(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i, aspect_index:last_index, :],dim=0) if aspect_index != last_index else spans_embedding[i][last_index]
                    # elif aspect_index == 0 and opinion_index == 0:
                    #     left = torch.mean(spans_embedding[i, 1:last_index, :], dim=0)
                    #     right = torch.mean(spans_embedding[i, 1:last_index, :], dim=0)
                    # else:
                    #     left = torch.mean(spans_embedding[i, 1:aspect_index, :], dim=0) if aspect_index != 1 else spans_embedding[i][0]
                    #     right = torch.mean(spans_embedding[i, aspect_index:last_index, :],dim=0) if aspect_index != last_index else spans_embedding[i][last_index]

                    aspect_rep = spans_embedding[i][aspect_index]
                    opinion_index = opinion_index if opinion_index!=0 else last_index
                    opinion_rep = spans_embedding[i][opinion_index]

                    # final_rep = torch.unsqueeze(torch.cat((aspect_rep+opinion_rep,left,right),dim=0),0)
                    final_rep = torch.unsqueeze(torch.cat((aspect_rep,opinion_rep),dim=0),0)

                    if input_rep == []:
                        input_rep = final_rep
                    else:
                        input_rep = torch.cat((input_rep, final_rep), dim=0)
            category_logits = self.category_classifier(input_rep)
            # category_logits = self.kan(input_rep)
            category_label = torch.cat(category_labels)

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

class imp_classifier(nn.Module):
    def __init__(self, embed_dim, num_filters, filter_sizes, output_dim, dropout):
        super(imp_classifier, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_text):
        # Permute to (batch_size, embed_dim, sequence_length)
        embedded_text = embedded_text.permute(0, 2, 1)

        conved = [nn.functional.relu(conv(embedded_text)) for conv in self.conv_layers]

        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        output = self.fc(cat)

        return output

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_classes, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, hidden_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, hidden_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, sequence_length - kernel_size + 1), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, num_filters), ...]
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes) * num_filters)
        logit = self.fc(x)  # (batch_size, num_classes)
        return x,logit

class Bert_Blend_CNN(torch.nn.Module):
  def __init__(self,args):
    super(Bert_Blend_CNN, self).__init__()

    self.linear = torch.nn.Linear(args.bert_feature_dim, 2)
    self.textcnn = TextCNN(args)
    self.cls_mult = nn.MultiheadAttention(embed_dim=args.bert_feature_dim, num_heads=4)
    self.cls_linear = nn.Linear(args.bert_feature_dim*12,args.bert_feature_dim)
  def forward(self, outputs):
    # input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    # 取每一层encode出来的向量
    # outputs.pooler_output: [bs, hidden_size]
    hidden_states = outputs.hidden_states # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
    cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) # [bs, 1, hidden]
    # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
    for i in range(2, 13):
      cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
    # cls_embeddings: [bs, encode_layer=12, hidden]
    cls_embeddings_tmp = cls_embeddings.clone()
    cls_embeddings_tmp = cls_embeddings_tmp.permute(1,0,2)
    imp_embedding =self.cls_mult(cls_embeddings_tmp,cls_embeddings_tmp,cls_embeddings_tmp)[0]
    imp_embedding = imp_embedding.permute(1,0,2)
    imp_embedding = self.cls_linear(imp_embedding.reshape(imp_embedding.size(0),-1))
    logits = self.textcnn(cls_embeddings)
    return logits,imp_embedding

class AttentionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, attention_dim, dropout_rate=0.5):
        super(AttentionCNN, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.attention_layer = nn.Linear(out_channels, attention_dim)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)  # 使用最大池化替代全局平均池化
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_layer = nn.Linear(out_channels, 2)

    def forward(self, x):
        # Convolutional layer
        conv_output = self.conv_layer(x.transpose(1, 2))

        # Attention mechanism
        attention_weights = torch.softmax(self.attention_layer(conv_output.permute(0, 2, 1)), dim=2)
        attended_features = torch.sum(attention_weights.permute(0,2,1) * conv_output, dim=2)

        pooled_output = self.max_pooling(attended_features.unsqueeze(2))

        # Dropout
        dropped_output = self.dropout(pooled_output.squeeze(2))

        # Fully connected layer
        output = self.fc_layer(dropped_output.view(dropped_output.size(0), -1))

        return output




'''
parameter()将一个不可训练的类型Tensor转换成可以
训练的类型parameter并将其绑定到这个module里面，
所以经过类型转换这个就变成了模型的一部分，成为了
模型中根据训练可以改动的参数了。使用这个函数的目
的也是想让某些变量在学习的过程中不断的修改其值以
达到最优化。
'''


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features,adj_matrix):
        # adj_matrix: 邻接矩阵, shape为 (batch_size, num_nodes, num_nodes)
        # node_features: 节点特征矩阵, shape为 (batch_size, num_nodes, in_features)
        # 输出节点的表示向量, shape为 (batch_size, num_nodes, out_features)
        num_nodes = adj_matrix.size(1)
        h = self.linear(node_features)
        # GCN公式：H' = A * H * W
        output = torch.bmm(adj_matrix, h)
        return output


# 打印形式是：GraphConvolution (输入特征 -> 输出特征)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.layer2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, node_features,adj_matrix):
        h = F.relu(self.layer1(node_features,adj_matrix))
        h = self.layer2(h,adj_matrix)
        return h


class CustomContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CustomContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 识别类别为1的样本
        class_1_indices = torch.nonzero(labels == 2).squeeze(1)
        class_1_indices = torch.cat((class_1_indices,class_1_indices),dim=0)

        # 如果存在类别为1的样本
        if len(class_1_indices) > 0:
            # 对类别为1的样本进行扰动（这里使用dropout）
            perturbed_features = F.dropout(features[class_1_indices], p=0.5)
            # 将扰动后的样本与原始特征拼接起来
            features = torch.cat((features, perturbed_features), dim=0)
            # 更新标签
            labels = torch.cat((labels, labels[class_1_indices]), dim=0)

        # 重新计算相似度
        similarities = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature

        # 生成对比掩码
        new_batch_size = features.size(0)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)[:new_batch_size]).float()

        # 计算对比损失
        numerator = torch.exp(similarities) * mask
        denominator = numerator.sum(dim=1, keepdim=True)
        log_prob = similarities - torch.log(denominator)
        loss = -log_prob.diag().mean()

        return loss





def eval_loss(gold_aspect_label,pred_aspect_label,reverse_gold_opinion_label, reverse_pred_opinion_label, spans_mask_tensor,
              exist_aspect,exist_opinion,imp_aspect_exist,imp_opinion_exist,args):
    # pos_weight = torch.tensor([args.binary_weight]).to(args.device)
    # imp_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    # aspect_loss
    aspect_spans_mask_tensor = spans_mask_tensor.view(-1) == 1
    pred_aspect_label_logits = pred_aspect_label.view(-1, pred_aspect_label.shape[-1])
    gold_aspect_effective_label = torch.where(aspect_spans_mask_tensor, gold_aspect_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(gold_aspect_label))
    aspect_loss = loss_function(pred_aspect_label_logits, gold_aspect_effective_label)

    #opinion_loss
    reverse_opinion_span_mask_tensor = spans_mask_tensor.view(-1) == 1
    reverse_pred_opinion_label_logits = reverse_pred_opinion_label.view(-1, reverse_pred_opinion_label.shape[-1])
    reverse_gold_opinion_effective_label = torch.where(reverse_opinion_span_mask_tensor,
                                                       reverse_gold_opinion_label.view(-1),
                                                       torch.tensor(loss_function.ignore_index).type_as(
                                                           reverse_gold_opinion_label))
    reverse_opinion_loss = loss_function(reverse_pred_opinion_label_logits, reverse_gold_opinion_effective_label)

    # exist_aspect = F.one_hot(exist_aspect, num_classes=2).float()
    # exist_opinion = F.one_hot(exist_opinion, num_classes=2).float()
    imp_aspect_loss = loss_function(imp_aspect_exist, exist_aspect)
    imp_opinion_loss = loss_function(imp_opinion_exist, exist_opinion)

    return aspect_loss,reverse_opinion_loss,imp_aspect_loss,imp_opinion_loss


def Loss(gold_aspect_label, pred_aspect_label, gold_opinion_label, pred_opinion_label, spans_mask_tensor, opinion_span_mask_tensor,
         reverse_gold_opinion_label, reverse_pred_opinion_label, reverse_gold_aspect_label, reverse_pred_aspect_label,
         cnn_spans_mask_tensor, reverse_aspect_span_mask_tensor, spans_embedding, related_spans_tensor,gold_category_label,
         pred_category_label, exist_aspect,exist_opinion,imp_aspect_exist,imp_opinion_exist,sentence_length,asp_rep,opi_rep,count,
        args):

    if cnn_spans_mask_tensor is not None:
        spans_mask_tensor = cnn_spans_mask_tensor
    # pos_weight = torch.tensor([args.binary_weight]).to(args.device)
    # imp_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    imp_loss_function = CustomContrastiveLoss()
    # Loss正向
    aspect_spans_mask_tensor = spans_mask_tensor.view(-1) == 1
    pred_aspect_label_logits = pred_aspect_label.view(-1, pred_aspect_label.shape[-1])
    gold_aspect_effective_label = torch.where(aspect_spans_mask_tensor, gold_aspect_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(gold_aspect_label))
    aspect_loss = loss_function(pred_aspect_label_logits, gold_aspect_effective_label)

    category_loss = torch.tensor(0)

    # pred_category_label_logits = pred_category_label.view(-1,pred_category_label.shape[-1])
    # gold_category_effective_label = torch.where(aspect_spans_mask_tensor, gold_category_label.view(-1),
    #                                           torch.tensor(loss_function.ignore_index).type_as(gold_category_label))
    # category_loss = 0
    # if pred_category_label != [] and gold_category_label != []:
    if count!=0:
        category_loss = loss_function(pred_category_label,gold_category_label)

    # pred_category_label_logits = pred_category_label.view(-1, pred_category_label.shape[-1])
    # gold_category_effective_label = torch.where(aspect_spans_mask_tensor, gold_category_label.view(-1),
    #                                           torch.tensor(loss_function.ignore_index).type_as(gold_category_label))
    # category_loss = loss_function(pred_category_label_logits, gold_category_effective_label)
    opinion_loss = 0
    if count!=0:
        opinion_span_mask_tensor = opinion_span_mask_tensor.view(-1) == 1
        pred_opinion_label_logits = pred_opinion_label.view(-1, pred_opinion_label.shape[-1])
        gold_opinion_effective_label = torch.where(opinion_span_mask_tensor, gold_opinion_label.view(-1),
                                                   torch.tensor(loss_function.ignore_index).type_as(gold_opinion_label))
        opinion_loss = loss_function(pred_opinion_label_logits, gold_opinion_effective_label)

        as_2_op_loss = aspect_loss + opinion_loss
    else:
        as_2_op_loss = aspect_loss


    # Loss反向
    reverse_opinion_span_mask_tensor = spans_mask_tensor.view(-1) == 1
    reverse_pred_opinion_label_logits = reverse_pred_opinion_label.view(-1, reverse_pred_opinion_label.shape[-1])
    reverse_gold_opinion_effective_label = torch.where(reverse_opinion_span_mask_tensor, reverse_gold_opinion_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(reverse_gold_opinion_label))
    reverse_opinion_loss = loss_function(reverse_pred_opinion_label_logits, reverse_gold_opinion_effective_label)

    if count != 0:
        reverse_aspect_span_mask_tensor = reverse_aspect_span_mask_tensor.view(-1) == 1
        reverse_pred_aspect_label_logits = reverse_pred_aspect_label.view(-1, reverse_pred_aspect_label.shape[-1])
        reverse_gold_aspect_effective_label = torch.where(reverse_aspect_span_mask_tensor, reverse_gold_aspect_label.view(-1),
                                                   torch.tensor(loss_function.ignore_index).type_as(reverse_gold_aspect_label))
        reverse_aspect_loss = loss_function(reverse_pred_aspect_label_logits, reverse_gold_aspect_effective_label)
        op_2_as_loss = reverse_opinion_loss + reverse_aspect_loss
    else:
        op_2_as_loss = reverse_opinion_loss
        # op_2_as_loss = reverse_aspect_loss
    # exist_aspect = F.one_hot(exist_aspect, num_classes=2).float()
    # exist_opinion = F.one_hot(exist_opinion, num_classes=2).float()
    imp_aspect_loss = loss_function(imp_aspect_exist, exist_aspect)
    imp_opinion_loss = loss_function(imp_opinion_exist, exist_opinion)

    # imp_asp_label_tensor = F.one_hot(imp_asp_label_tensor, num_classes=2).float()
    # imp_asp_loss = imp_criterion(aspect_imp_logits, imp_asp_label_tensor)
    # imp_opi_label_tensor = F.one_hot(imp_opi_label_tensor, num_classes=2).float()
    # imp_opi_loss = imp_criterion(opinion_imp_logits, imp_opi_label_tensor)
    # imp_opi_loss = loss_function(aspect_imp_logits,imp_asp_label_tensor.view(-1))
    # F.one_hot(imp_asp_label_tensor, num_classes=2)

    # aspect_polarity_loss = loss_function(aspect_imp_polarity_logits,aspect_polarity_label_tensor.view(-1))
    # opinion_polarity_loss = loss_function(opinion_imp_polarity_logits,opinion_polarity_label_tensor.view(-1))
    # prob_loss = L1(opinion_soft,forward_softmax)+L1(aspect_soft,reverse_softmax)
    con_asp_loss = imp_loss_function(asp_rep,exist_aspect)
    con_opi_loss = imp_loss_function(opi_rep,exist_opinion)


    if args.kl_loss:
        kl_loss = shape_span_embedding(args, spans_embedding, spans_embedding, related_spans_tensor, spans_mask_tensor,sentence_length)
        # loss = as_2_op_loss + op_2_as_loss + kl_loss
        # loss = as_2_op_loss + op_2_as_loss + category_loss + args.kl_loss_weight * kl_loss + imp_aspect_loss + imp_opinion_loss
        loss = as_2_op_loss + op_2_as_loss + category_loss + imp_aspect_loss + \
              imp_opinion_loss + args.kl_loss_weight * kl_loss+ con_asp_loss + con_opi_loss
    else:
        loss = as_2_op_loss + op_2_as_loss + imp_aspect_loss + imp_opinion_loss+ category_loss + con_asp_loss + con_opi_loss
        kl_loss = 0


    return loss,  args.kl_loss_weight * kl_loss,as_2_op_loss,op_2_as_loss,imp_aspect_loss,imp_opinion_loss,category_loss,con_asp_loss,con_opi_loss







def compute_con_loss(args,aspect_tensor, opinion_tensor):
    con_loss = 0
    all_cos = []
    if aspect_tensor is None or opinion_tensor is None:
        return 0
    for asp_index_out,asp_out in enumerate(aspect_tensor):
        asp_out = asp_out.expand_as(aspect_tensor)
        cos_dis = torch.cosine_similarity(asp_out, aspect_tensor, dim=1)
        all_cos.append(cos_dis)

    all_cos = torch.stack(all_cos)
    all_dis = []
    for index in range(len(aspect_tensor)):
        asp_rep = aspect_tensor[index].expand_as(opinion_tensor)
        dis = torch.cosine_similarity(asp_rep,opinion_tensor,dim=1)
        all_dis.append(dis)
    all_dis = torch.stack(all_dis)

    all_cos = all_cos / args.temp
    all_dis = all_dis / args.temp

    all_cos = torch.exp(all_cos)
    all_dis = torch.exp(all_dis)

    row_sum = []
    for i in range(len(all_dis)):
        row_sum.append(sum(all_dis[i]))

    for i in range(len(aspect_tensor)):
        n_i = len(aspect_tensor)-1
        inner_sum = 0
        for j in range(len(aspect_tensor)):
            if i!=j:
                inner_sum = inner_sum + torch.log(all_cos[i][j]/row_sum[i])
        if n_i!=0:
            con_loss += (inner_sum / (-n_i))

    return con_loss/10
def shape_con_loss(args, spans_embedding, gold_aspect_label, spans_mask_tensor, sentence_length):
    aspect_tensor = None
    opinion_tensor = None
    input_size = spans_embedding.size()

    for i,value_list in enumerate(gold_aspect_label):
        last_index = torch.sum(spans_mask_tensor[i]).item()-1
        for index,value in enumerate(value_list):
            if value.item() == 1 and index!=0:
                if aspect_tensor == None:
                    aspect_tensor = spans_embedding[i,index,:].unsqueeze(0)
                else:
                    aspect_tensor = torch.cat((aspect_tensor,spans_embedding[i,index,:].unsqueeze(0)),dim=0)
            elif value.item() == 2 and index!=last_index:
                if opinion_tensor == None:
                    opinion_tensor = spans_embedding[i, index, :].unsqueeze(0)
                else:
                    opinion_tensor = torch.cat((opinion_tensor, spans_embedding[i, index, :].unsqueeze(0)), dim=0)
            else:
                continue
    con_loss =compute_con_loss(args,aspect_tensor,opinion_tensor)+ compute_con_loss(args,opinion_tensor,aspect_tensor)
    return con_loss


def shape_span_embedding(args, p, q, pad_mask, span_mask,sentence_length):
    kl_loss = 0
    input_size = p.size()
    random.seed(args.RANDOM_SEED)
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
        if (p_loss + q_loss)>0:
            total_loss = math.log(1+5/((p_loss + q_loss) / 2))
        if (p_loss + q_loss) <= 0:
            total_loss = 0
            if (p_loss + q_loss)<0:
                print(p_loss + q_loss)
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
