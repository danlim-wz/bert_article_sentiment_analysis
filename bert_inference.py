import torch
from pytorch_transformers import *
import numpy as np
import os
from flask import Flask, request, make_response

app = Flask(__name__)

#load model and tokenizer
config = BertConfig()
model = BertForSequenceClassification(config)
model_state_dict = "./bertweight.pth"
model.load_state_dict(torch.load(model_state_dict))
model.cuda().eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

@app.route("/", methods=['POST'])
def inference():
    input_file = request.files['input']
    data = "[CLS] " + str(input_file.read()) + " [SEP]"

    #tokenize and slice input data - set last token to "[SEP]" if sentence length > 512
    token_text = []
    tokenized_text = tokenizer.tokenize(data)
    tokenized_text = tokenized_text[:512]
    tokenized_text[511] = "[SEP]"
    token_text.append(tokenized_text)

    # #convert to indices and send to GPU memory
    indexed_text = [tokenizer.convert_tokens_to_ids(text) for text in token_text]
    indexed_text = torch.tensor(indexed_text).cuda()

    #make prediction
    with torch.no_grad():
        logits = model(indexed_text)
    logits = [p.cpu().squeeze() for p in logits]
    softmax = torch.nn.Softmax(dim=0)
    logits = softmax(logits[0]).numpy()

    #output result
    pos_neg = np.argmax(logits, axis=0)
    if pos_neg == 0:
        return make_response('Overall sentiment: ' + 'positive ' + str(logits[0]))
    else:
        return make_response('Overall sentiment: ' + 'negative ' + str(logits[1]))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
    
