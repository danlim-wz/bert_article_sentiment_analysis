import torch
from pytorch_transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tunable parameters
MAX_LEN = 256
batch_size = 6
epochs = 4
learning_rate = 2e-5

#tokenizer and model objects
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.eval().to(device)

#import & process data
input_path = '/path/to/input/dataset'
train_data = []
training_label = []
for file in os.listdir(input_path):
    score, ext = os.path.splitext(file)
    score = score.split('_')[1]
    if int(score) > 5:
        label = 1 #postive
    else:
        label = 0 #negative
    training_label.append(label)
    with open(os.path.join(input_path,file)) as f:
        input_text ="[CLS] "+ f.read() + " [SEP]"
        train_data.append(input_text)

tokenized_text = [tokenizer.tokenize(train_sent) for train_sent in train_data]
indexed_text = pad_sequences([tokenizer.convert_tokens_to_ids(text) for text in tokenized_text],
                 maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

#manually change last token to [SEP] since sentences may be longer than MAX_LEN
for sentence in indexed_text:
    if sentence[MAX_LEN-1] != 0:
        sentence[MAX_LEN-1] = 102 #sep_index is 102

#attention masks
attention_masks = []
for id in indexed_text:
    mask = [int(i>0) for i in id]
    attention_masks.append(mask)

#split into train/validation sets
split = int(0.8*len(indexed_text))
train_data = indexed_text[:split]
train_label = training_label[:split]
validation_data = indexed_text[split:]
validation_label = training_label[split:]
train_masks = attention_masks[:split]
validation_masks = attention_masks[split:]

# convert to torch tensors
train_data = torch.tensor(train_data)
train_label = torch.tensor(train_label)
validation_data = torch.tensor(validation_data)
validation_label = torch.tensor(validation_label)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#create dataloader
train_data = TensorDataset(train_data, train_masks, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_data, validation_masks, validation_label)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


#create optimizer
param_optimizer = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer]}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate)

#train/validation loop
for k in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    tr_steps = 0
    #train loop
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        loss, _ = model(b_input, token_type_ids=None, attention_mask=b_input_mask,
                    labels=b_labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        tr_steps += 1
    print("Training loss: {}".format(tr_loss/tr_steps))

    #validation loop
    model.eval()
    val_steps = 0
    val_acc = 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input, token_type_ids=None, attention_mask=b_input_mask)
        logits = [p.cpu().numpy().squeeze() for p in logits]
        label_ids = b_labels.cpu().numpy()
        logits = np.argmax(logits[0], axis=1).flatten()
        label_ids = label_ids.flatten()
        val_acc += np.sum(logits == label_ids)/len(label_ids) 
        val_steps += 1
    print("Validation Accuracy: {}".format(val_acc/val_steps))
    torch.save(model.state_dict(),'/path/to/output/directory'+str(k)+'.pth')

