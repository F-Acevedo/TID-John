import torch
from transformers import BertTokenizer, BertModel
import numpy as np


model_version = 'allenai/scibert_scivocab_uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version ,output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
