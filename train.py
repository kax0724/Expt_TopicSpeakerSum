import os
import sys
sys.path.append('../../transformers/src')

import pandas as pd
import numpy as np

from transformers import AdamW, pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import torch

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch_device:",torch_device)

model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
# batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest').to(torch_device)
if torch_device == 'cuda':
    model = torch.nn.DataParallel(model).to(torch_device)
else:
    model = model.to(torch_device)

corpus_dir = "/Users/kaima/PycharmProjects/Expt_TopicSpeakerSum/TopSpeakSum/data/corpus"
df_train = pd.read_table(os.path.join(corpus_dir,"train.tsv"), index_col=0)

dialogues = list(df_train['dialogue'][:4].values)
summaries = list(df_train['summary'][:4].values)

batch = tokenizer.prepare_seq2seq_batch(dialogues, truncation=True, max_length=256).to(torch_device)

model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
encoding = tokenizer(dialogues, return_tensors='pt', padding=True, truncation=True)
# input_ids = encoding['input_ids'].to(torch_device)
# attention_mask = encoding['attention_mask'].to(torch_device)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

outputs = model(input_ids, attention_mask=attention_mask)

loss = outputs.loss
loss.backward()
optimizer.step()

