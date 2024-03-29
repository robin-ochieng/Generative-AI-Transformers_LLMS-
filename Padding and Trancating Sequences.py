#Putting it all together
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

model_inputs = tokenizer(sequences)

sequences2 = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequences2)

sequences3 = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences3)

# Will pad the sequences up to the maximum sequence length
model_inputs1 = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs2 = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs3 = tokenizer(sequences, padding="max_length", max_length=14)

#It can also truncate sequences to a maximum length if they are longer than the model 
sequences_2 = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences_2, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences_2, max_length=8, truncation=True)


sequences_3 = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
# Returns PyTorch tensors
model_inputs_1 = tokenizer(sequences_3, padding=True, return_tensors="pt")
# Returns TensorFlow tensors
model_inputs_2 = tokenizer(sequences_3, padding=True, return_tensors="tf")
# Returns NumPy arrays
model_inputs_3 = tokenizer(sequences_3, padding=True, return_tensors="np")

print(model_inputs_1["input_ids"])
print(model_inputs_2)
print(model_inputs_3)

sequences_3_ = 'I have been waiting for a'
tokens = tokenizer.tokenize(sequences_3_)
ids_1 = tokenizer.convert_tokens_to_ids(tokens) 
model_inputs_4 = tokenizer(sequences_3_) 
print(ids_1)

print(tokenizer(sequences_3_)['attention_mask'])

print(tokenizer.decode(model_inputs_4['input_ids']))
print(tokenizer.decode(ids_1))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)

