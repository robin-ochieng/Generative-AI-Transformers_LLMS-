#Preprocessing with a tokenizer
import torch
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

tokenized_text1 = "Let’s do tokenization! Jim's Henson was a puppeteer've".split()
print(tokenized_text1)

tokenized_text = "Let’s do tokenization! Jim's Henson was a puppeteer've"
print(tokenized_text)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenized_text2 = tokenizer.tokenize(tokenized_text)
print(tokenized_text2)

from transformers import AutoTokenizer
tokenizer1 = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_text3 = tokenizer1("Using a Transformer network is simple")
print(tokenized_text3)

ids = tokenizer.convert_tokens_to_ids(tokenized_text2)
print(ids)

decoded_string = tokenizer.decode([2421, 787, 188, 1202, 22559, 2734, 106, 3104, 112, 188, 1124, 15703, 1108, 170, 16797, 8284, 112, 1396])
print(decoded_string)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence = "I will be worth 100,000 Us Dollars by 2034"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
print("input_ids:", input_ids)
outputs = model(input_ids)
print('Logits:', outputs.logits)


tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(input_ids)
print(tokenized_inputs["input_ids"])





checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence = "I will be worth 100,000 Us Dollars by 2034"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
batched_ids = [ids, ids]
print(batched_ids)
input_ids = torch.tensor(batched_ids)
print("input_ids:", input_ids)
outputs = model(input_ids)
print('Logits:', outputs.logits)


padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],]

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [[200, 200, 200],[200, 200, tokenizer.pad_token_id],]
model1 = model(torch.tensor(sequence1_ids))
model2 = model(torch.tensor(sequence2_ids))
model3 = model(torch.tensor(batched_ids))
print(model1.logits)
print(model2.logits)
print(model3.logits)

#Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to
batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id],]
attention_mask = [[1, 1, 1], [1, 1, 0],]
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)



checkpoint1 = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer1 = AutoTokenizer.from_pretrained(checkpoint1)
model_1 = AutoModelForSequenceClassification.from_pretrained(checkpoint1)
sequence1 = [["I've been waiting for a HuggingFace course my whole life."], ["I hate this so much!",]]
tokens1 = tokenizer.tokenize(sequence1)
ids1 = tokenizer.convert_tokens_to_ids(tokens1)
print(ids1)
input_ids1 = torch.tensor(ids1)
print("input_ids:", input_ids1)
outputs = model(input_ids)
print('Logits:', outputs.logits)

