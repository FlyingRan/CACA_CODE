import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
# from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .Attention import Attention, Intermediate, Output, Dim_Four_Attention, masked_softmax
from .data_BIO_loader import sentiment2id, validity2id,categories,id2category
from allennlp.nn.util import batched_index_select, batched_span_select
import random
import math

def stage_2_features_generation(bert_feature, attention_mask, spans, span_mask, spans_embedding, spans_aspect_tensor,
                                is_aspect,spans_opinion_tensor=None):
    # 对输入的aspect信息进行处理，去除掉无效的aspect span
    all_span_aspect_tensor = None
    all_span_opinion_tensor = None
    all_bert_embedding = None
    all_attention_mask = None
    all_spans_embedding = None
    all_span_mask = None
    spans_aspect_tensor_spilt = torch.chunk(spans_aspect_tensor, spans_aspect_tensor.shape[0], dim=0)
    flag = 0
    for i, spans_aspect_tensor_unspilt in enumerate(spans_aspect_tensor_spilt):
        test = spans_aspect_tensor_unspilt.squeeze()
        batch_num = spans_aspect_tensor_unspilt.squeeze(0)[0]
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        if (test[1] == test[2] == -1) or (test[1] == test[2] == 0):
            flag = 1
        else:
            span_index_start = torch.where(spans[batch_num, :, 0] == spans_aspect_tensor_unspilt.squeeze()[1],
                                           spans[batch_num, :, 1], torch.tensor(-1).type_as(spans))
            span_index_end = torch.where(span_index_start == spans_aspect_tensor_unspilt.squeeze()[2], span_index_start,
                                         torch.tensor(-1).type_as(spans))
            span_index = torch.nonzero((span_index_end > -1), as_tuple=False).squeeze(0)
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
                aspect_span_embedding_unspilt = spans_embedding[batch_num,  torch.tensor([len(spans[batch_num])-1]), :].unsqueeze(0)
            flag = 0
        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)
        if all_span_aspect_tensor is None:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = spans_opinion_tensor_unspilt
            all_span_aspect_tensor = aspect_span_embedding_unspilt
            all_bert_embedding = bert_feature_unspilt
            all_attention_mask = attention_mask_unspilt
            all_spans_embedding = spans_embedding_unspilt
            all_span_mask = span_mask_unspilt
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)
            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
           all_spans_embedding, all_span_mask

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
        self.category_classifier = nn.Sequential(
            nn.Linear(768 * 2, len(categories))
        )
        self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)
            # self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(sentiment2id))
        self.categories_classification_opinion = nn.Linear(args.bert_feature_dim, len(categories))
        self.imp_asp_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.hidden_size, 2)
        )
        self.imp_opi_classifier = nn.Sequential(
            nn.Dropout(args.drop_out),
            nn.Linear(args.hidden_size, 2)
        )
    def forward(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor, pooler_output,sentence_length):
        # 调用 span_generator 方法生成 spans_embedding 和 features_mask_tensor 两个变量。
        # 其中 spans_embedding 表示每个 span 的嵌入表示，features_mask_tensor 表示每个 span 是否被标记为有效的情感单元。
        spans_embedding, features_mask_tensor = self.span_generator(input_bert_features, attention_mask, spans,
                                                                    span_mask, related_spans_tensor,pooler_output, sentence_length)

        span_embedding_1 = torch.clone(spans_embedding)
        for forward_1_decoder in self.forward_1_decoders:
            forward_layer_output, forward_intermediate_output = forward_1_decoder(span_embedding_1)
            span_embedding_1 = forward_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 aspect 的预测结果，
        # 使用 sentiment_classification_aspect 层将其映射到具体的情感极性标签上，得到 class_logits_aspect。
        class_logits_aspect = self.sentiment_classification_aspect(span_embedding_1)
        class_logits_category = self.categories_classification_opinion(span_embedding_1)
        # 复制 spans_embedding 得到 span_embedding_1 和 span_embedding_2 两个变量，
        # 然后分别将它们输入到 forward_1_decoders 和 reverse_1_decoders 中进行解码。
        span_embedding_2 = torch.clone(spans_embedding)
        for reverse_1_decoder in self.reverse_1_decoders:
            reverse_layer_output, reverse_intermediate_output = reverse_1_decoder(span_embedding_2)
            span_embedding_2 = reverse_layer_output
        # 同样用 sentiment_classification_opinion 层将其映射到具体的情感极性标签上，得到 class_logits_opinion。
        class_logits_opinion = self.sentiment_classification_opinion(span_embedding_2)

        # pred_aspect = torch.argmax(F.softmax(class_logits_aspect,dim=2),dim=2)

        # aspect_rep = []
        # opinion_rep = []
        category_label = []
        categories_logits = []
        # if torch.nonzero(pred_aspect, as_tuple=False).size(0) != 0:
        #     pred_opinion = torch.argmax(F.softmax(class_logits_opinion, dim=2), dim=2)
        #     if torch.nonzero(pred_opinion, as_tuple=False).size(0) != 0:
        #         opinion_span = torch.chunk(torch.nonzero(pred_opinion, as_tuple=False),
        #                                    torch.nonzero(pred_opinion, as_tuple=False).shape[0], dim=0)
        #         aspect_span = torch.chunk(torch.nonzero(pred_aspect, as_tuple=False),
        #                                    torch.nonzero(pred_aspect, as_tuple=False).shape[0], dim=0)
        #
        #         real_pair = torch.tensor(self.cal_real(sentence_length[0][3],spans[0]))
        #         pred_pair  = torch.tensor([[aspect_span[i][0][1], opinion_span[j][0][1]] for i in range(len(aspect_span)) for j in range(len(opinion_span))])
        #         for aspect_idx in aspect_span:
        #             aspect_rep.append(spans_embedding[aspect_idx[0][0], aspect_idx[0][1]])
        #         aspect_rep = torch.stack(aspect_rep)
        #         for opinion_idx in opinion_span:
        #             opinion_rep.append(spans_embedding[opinion_idx[0][0], opinion_idx[0][1]])
        #         opinion_rep = torch.stack(opinion_rep)
        #         category_label = self.create_category_label(real_pair,pred_pair,sentence_length[0][4])
        #         cartesian_products = self.cartesian_product(aspect_rep,opinion_rep)
        #         categories_logits = self.category_classifier(cartesian_products)
        return class_logits_aspect, class_logits_opinion, spans_embedding, span_embedding_1, span_embedding_2, \
               features_mask_tensor,categories_logits,category_label

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
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))

            if self.args.span_generation == "Max":
                spans_width_start_end_max = spans_width_start_end_embedding.max(2)
                spans_embedding = spans_width_start_end_max[0]
            else:
                spans_width_start_end_mean = spans_width_start_end_embedding.mean(dim=2, keepdim=True).squeeze(-2)
                spans_embedding = spans_width_start_end_mean
            for i in range(len(sentence_length)):
                spans_embedding[i][0] = pooler_output[i].squeeze()
                spans_embedding[i][sentence_length[i][2]-1] = pooler_output[i].squeeze()
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
    def forward(self, hidden_embedding, masks, encoder_embedding):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, hidden_dim, encoder_dim)
        # 计算掩码，将 ~masks 乘以 -1e9，使得无意义位置对应的值趋近于负无穷
        masks = (~masks) * -1e9
        # 根据掩码的维度信息确定 attention_masks 的形状
        if masks.dim() == 3:
            attention_masks = masks[:, None, :, :]
        elif masks.dim() == 2:
            if self.mask_for_encoder:
                attention_masks = masks[:, None, None, :]
            else:
                attention_masks = masks[:, None, :, None]
        if self.mask_for_encoder:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                       encoder_hidden_states=encoder_embedding,
                                                       encoder_attention_mask=attention_masks)
        else:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
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

    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor):
        # 接收输入参数 aspect_spans_embedding（aspect 的嵌入表示）、
        # aspect_span_mask（aspect 的掩码）和 spans_aspect_tensor（aspect 的索引位置）
        '''aspect---> opinion 方向'''
        # 将aspect_spans_embedding输入到解码器循环解码
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding,
                                                                            aspect_span_mask,
                                                                            spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 opinion 的预测结果，
        # 使用 opinion_docoder2class 层将其映射到具体的情感极性标签上，得到 opinion_class_logits。
        opinion_class_logits = self.opinion_docoder2class(aspect_spans_embedding)
        return opinion_class_logits, opinion_attention

class Step_3_categories(torch.nn.Module):
    def __init__(self, args):
        super(Step_3_categories, self).__init__()
        self.args = args
        self.fc = nn.Linear(768*2, len(categories))
        self.dropout = nn.Dropout(0.5)
        self.decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
    def forward(self,spans_embedding,bert_spans_tensor,pairs,category_label=None):
        if category_label == None:
            category_logits = self.fc(pairs)
            # category_logits = self.dropout(category_logits)
            return category_logits,[]
        else:
            real_pair = self.cal_real(pairs,bert_spans_tensor[0])
            input_rep = []
            for pair in real_pair:
                aspect_index = int(pair[0])
                opinion_index = int(pair[1])
                aspect_rep = spans_embedding[0][aspect_index]
                opinion_rep = spans_embedding[0][opinion_index]
                final_rep = torch.unsqueeze(torch.cat((aspect_rep, opinion_rep),dim=0),0)
                if input_rep == []:
                    input_rep = final_rep
                else:
                    input_rep= torch.cat((input_rep,final_rep),dim=0)
            category_logits = self.fc(input_rep)
            # category_logits = self.dropout(category_logits)
            category_label = category_label
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

    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor):
        '''opinion---> aspect 方向'''
        # 将reverse_spans_embedding输入到解码器循环解码
        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding,
                                                                                 reverse_span_mask,
                                                                                 all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        # 将最后一个解码器的输出结果作为情感极性关于 aspect 的预测结果，
        # 使用 aspect_docoder2class 层将其映射到具体的情感极性标签上，得到 aspect_class_logits。
        aspect_class_logits = self.aspect_docoder2class(reverse_spans_embedding)
        return aspect_class_logits, aspect_attention



def Loss(gold_aspect_label, pred_aspect_label, gold_opinion_label, pred_opinion_label, spans_mask_tensor, opinion_span_mask_tensor,
         reverse_gold_opinion_label, reverse_pred_opinion_label, reverse_gold_aspect_label, reverse_pred_aspect_label,
         cnn_spans_mask_tensor, reverse_aspect_span_mask_tensor, spans_embedding, related_spans_tensor,gold_category_label,
         pred_category_label,args):
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    if cnn_spans_mask_tensor is not None:
        spans_mask_tensor = cnn_spans_mask_tensor

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

    if args.kl_loss:
        kl_loss = shape_span_embedding(args, spans_embedding, spans_embedding, related_spans_tensor, spans_mask_tensor)
        # loss = as_2_op_loss + op_2_as_loss + kl_loss
        loss = as_2_op_loss + op_2_as_loss + args.kl_loss_weight * kl_loss
    else:
        loss = as_2_op_loss + op_2_as_loss
        kl_loss = 0
    return loss, args.kl_loss_weight * kl_loss

def shape_span_embedding(args, p, q, pad_mask, span_mask):
    kl_loss = 0
    input_size = p.size()
    assert input_size == q.size()
    for i in range(input_size[0]):
        span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
        if len(span_mask_index.shape) == 0:
            return 0
        flag = True
        while flag:
            lucky_squence = random.choice(span_mask_index)
            if lucky_squence < pad_mask.size()[1]-1:
                flag = False
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

def compute_kl_loss(args, p, q, pad_mask=None):
    if args.kl_loss_mode == "KLLoss":
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill(pad_mask, 0.)
            q_loss.masked_fill(pad_mask, 0.)
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
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
