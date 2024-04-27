
# To apply DeBERTa to your existing code, you need to make two changes to your code,
# 1. change your model to consume DeBERTa as the encoder
from DeBERTa import deberta
import torch
import torch
from DeBERTa import deberta
from transformers import AutoConfig, AutoModel,AutoTokenizer




# 2. Change your tokenizer with the tokenizer built-in DeBERta
model_path = "pretrained_models/deberta-v3/"
config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)

tokenizers = AutoTokenizer.from_pretrained(model_path)
data=['this is first sushi unresponsive','this is second sentence']
max_seq_len = 512
tokens = tokenizers.tokenize("unresponsive",is_split_into_words=True)
inputs = tokenizers(data,padding=True, truncation=True, return_tensors="pt",is_split_into_words=True)
# Truncate long sequence
# tokens = tokens[:max_seq_len -2]
print(inputs)
# Add special tokens to the `tokens`
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# features = {
# 'input_ids': torch.tensor(input_ids, dtype=torch.int),
# 'input_mask': torch.tensor(input_mask, dtype=torch.int)
# }
