import json
import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ques = [ "nose",
#     "left_eye", "right_eye",
#     "left_ear", "right_ear",
#     "left_shoulder", "right_shoulder",
#     "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist",
#     "left_hip", "right_hip",
#     "left_knee", "right_knee",
#     "left_ankle", "right_ankle"]
# token = []
# for word in ques:
#     token.extend(word.split('_'))
# ques = list(set(token))
# print(ques)
ques = 'che tang mig is a handsome boy'
ques = '[CLS] ' + ques
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# tokenized_ques = []
print(ques)
ques = tokenizer.tokenize(ques)
print(ques)
# for que in ques:
#     tokenized_ques.extend(tokenizer.tokenize(que))
# print(tokenized_ques)
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_ques)
# print(indexed_tokens)
# # input_mask = [1] * len(indexed_tokens)
# token_tensor = torch.tensor([indexed_tokens]).to('cuda')
# model.eval().to('cuda')
# encoded_layers, _ = model(token_tensor)
# q_embeds = []
# q_embeds.append(np.array(encoded_layers[-1].detach().cpu().squeeze()).tolist())
# # print(q_embeds[0].shape)
# print(len(q_embeds))
# # print(encoded_layers.shape)
# word_dict = {}
# for i, word_embed in enumerate(q_embeds[0]):
#     word_dict[ques[i]] = word_embed
# with open("./word_dict.json", "r") as f:
#     wod = json.load(f)
# print(len(wod))
# print(wod['right'])
# print(len(wod['right']))