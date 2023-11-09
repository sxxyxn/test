import re
import pandas as pd
import numpy
from transformers import AutoTokenizer,AutoModel
import torch
from konlpy.tag import Okt
import tables

model_path = 'KoDiffCSE-RoBERTa'

data1 = pd.read_table('naver_shopping.txt',names=['ratings','review'])
data2 = pd.read_table('ratings_train.txt')
data3 = pd.read_table('ratings_test.txt')

df = pd.concat([data1['review'],data2['document'],data3['document']],axis=0,ignore_index=True)
df.columns = ['text']

model = AutoModel.from_pretrained(model_path).to('cuda')
tok = AutoTokenizer.from_pretrained(model_path)

def vectorize(text):
    inputs = tok(text, padding=True, truncation=True, return_tensor="pt")
    inputs = {k:v.to('cuda') for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy()

num_len = len(str(len(df)))

df['vec'] = df['text'].apply(vectorize)
df['proc_num'] = [f'{i:0{num_len}d}' for i in range(1,len(df)+1)]

df.to_hdf('vector.h5',key='df',mode='w')