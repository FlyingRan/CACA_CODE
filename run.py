import argparse
from log import logger
from main import train
def main():
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument('--model_dir', type=str, default="savemodels/", help='model path prefix')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--init_model", default="pretrained_models/deberta-v3", type=str, required=False,help="Initial model.bert-base-uncased")
    parser.add_argument("--init_vocab", default="pretrained_models/deberta-v3", type=str, required=False,help="Initial vocab.")

    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=False, action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=150, type=int,help="The maximum total input sequence length after WordPiece tokenization. "
                                                                       "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--drop_out", type=int, default=0.5, help="")
    parser.add_argument('--special_token', default='[N]')
    parser.add_argument("--max_span_length", type=int, default=12, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument('--epochs', type=int, default=250, help='training epoch number')
    parser.add_argument("--train_batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=44, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset", default="laptop", type=str, choices=["restaurant", "laptop","phone","laptop14"],help="specify the dataset")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # 选择Span的合成方式
    parser.add_argument('--span_generation', type=str, default="Start_end_minus_plus", choices=["Start_end", "Max", "Average", "CNN", "ATT","Start_end_minus_plus","average_max"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")
    # 是否对相关span添加分离Loss
    parser.add_argument("--kl_loss", default=False)
    parser.add_argument("--kl_loss_weight", type=int, default=1, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')
    parser.add_argument("--binary_weight", type=int, default=4, help="weight of the binary loss")

    parser.add_argument("--temp", type=int, default=0.1, help="temperature")
    parser.add_argument("--con_weight", type=int, default=0.1, help="con_weight")
    # 是否使用测试中的筛选算法
    parser.add_argument('--Filter_Strategy',  default=True, help='是否使用筛选算法去除冲突三元组')
    parser.add_argument('--filter_a', type=float, default=0.95,help='筛选的阈值')
    parser.add_argument('--alpha', type=float, default=0.5,help='alpha')
    parser.add_argument('--beta', type=float, default=0.5,help='beta')
    parser.add_argument('--lambda', type=float, default=0.7,help='lambda')
    parser.add_argument('--tem', type=float, default=0.07,help='tem')
    # 选择Cross Attention中ATT块的个数
    parser.add_argument("--block_num", type=int, default=1, help="number of block")
    parser.add_argument("--output_path", default='triples.json')
    parser.add_argument("--whether_filter",default=False)
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
