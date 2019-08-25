import torch
from pytorch_transformers import *
import numpy as np
import os

#load model and tokenizer
config = BertConfig()
model = BertForSequenceClassification(config)
model_state_dict = "./bertweight.pth"
model.load_state_dict(torch.load(model_state_dict))
model.cuda().eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#get test input data
articles = ['HKarticle.txt','WHOarticle.txt','DENarticle.txt']
token_text = []
for article in articles:
    with open(os.path.join('/path/to/folder', article)) as f:
        data = "[CLS] " + f.read() + " [SEP]"

    #tokenize and slice input data
    tokenized_text = tokenizer.tokenize(data)
    tokenized_text = tokenized_text[:512]
    tokenized_text[511] = "[SEP]"
    token_text.append(tokenized_text)

#convert to indices and send to GPU memory
indexed_text = [tokenizer.convert_tokens_to_ids(text) for text in token_text]
indexed_text = torch.tensor(indexed_text).cuda()


with torch.no_grad():
    logits = model(indexed_text)

logits = [p.cpu().squeeze() for p in logits]
softmax = torch.nn.Softmax(dim=1)
logits = softmax(logits[0]).numpy()

for item in logits:
    pos_neg = np.argmax(item, axis=0)
    if pos_neg == 0:
        print('postive({0:.2f})'.format(item[0]))
    else:
        print('negative({0:.2f})'.format(item[1]))
