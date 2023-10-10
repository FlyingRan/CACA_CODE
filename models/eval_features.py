

def unbatch_data(pred_data):
    stage1_pred = []
    stage1_pred_sentiment = []
    stage1_pred_sentiment_logits = []
    if len(pred_data) == 7:
        stage1_pred_category= []
    stage2_pred = []
    stage2_pred_sentiment_logits = []
    imp_results = []

    for i in range(len(pred_data[0])):
        pred_stage1_result_tolist = pred_data[0][i].tolist()
        pred_stage1_result_sentiment_tolist = pred_data[1][i].tolist()
        pred_stage1_sentiment_logit_tolist = pred_data[2][i].tolist()

        pred_stage2_result_tolist = pred_data[3][i].tolist()
        pred_stage2_sentiment_logit_tolist = pred_data[4][i].tolist()

        imp_result_tolist = pred_data[5][i]
        if len(pred_data) == 7:
            pred_stage1_category_logit_tolist = pred_data[6][i]
        # test
        if len(pred_stage1_result_tolist) != len(pred_stage2_result_tolist):
            raise IndexError('预测的stage1和stage2序列数不相等')
        for j in range(len(pred_stage1_result_sentiment_tolist)):
            pred_stage1_per_sent, pred_stage2_per_sent, pred_stage2_sentiment_logit_per_sent = [], [], []
            pred_category_result = []
            stage1_pred_sentiment.append(pred_stage1_result_sentiment_tolist[j])
            stage1_pred_sentiment_logits.append(pred_stage1_sentiment_logit_tolist[j])
            imp_list_result = []
            # if len(pred_data) == 6:
            #     stage1_pred_category.append(pred_stage1_category_logit_tolist[j])
            for k2, pred_span in enumerate(pred_stage1_result_tolist):
                if pred_span[0] == j:
                    pred_stage1_per_sent.append(pred_span)
                    pred_stage2_per_sent.append(pred_stage2_result_tolist[k2])
                    pred_stage2_sentiment_logit_per_sent.append(pred_stage2_sentiment_logit_tolist[k2])
            if len(pred_data) == 7:
                for m,pred_category in enumerate(pred_stage1_category_logit_tolist):
                    if pred_category[3] == j:
                        pred_category_result.append(pred_category)
                stage1_pred_category.append(pred_category_result)

            for n, imp_result in enumerate(imp_result_tolist):
                if imp_result[0]==j:
                    imp_list_result.append(imp_result)
            imp_results.append(imp_list_result)
            stage1_pred.append(pred_stage1_per_sent)
            stage2_pred.append(pred_stage2_per_sent)
            stage2_pred_sentiment_logits.append(pred_stage2_sentiment_logit_per_sent)
    if len(pred_data) == 7:
        pred_result = (stage1_pred, stage1_pred_sentiment, stage1_pred_sentiment_logits, stage2_pred,
                       stage2_pred_sentiment_logits,imp_results,stage1_pred_category)
    else:
        pred_result = (stage1_pred, stage1_pred_sentiment, stage1_pred_sentiment_logits, stage2_pred,
                   stage2_pred_sentiment_logits,imp_results)
    return pred_result