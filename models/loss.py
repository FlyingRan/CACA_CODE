import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
# from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .cross_attention import Attention, Intermediate, Output, Dim_Four_Attention, masked_softmax,implict_ATT,SelfAttention
from .dataloader import sentiment2id, validity2id
from allennlp.nn.util import batched_index_select, batched_span_select
from .kan import *
import random
import math
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