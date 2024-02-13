import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the checkpoint and load the tokenizer and model
checkpoint1 = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer1 = AutoTokenizer.from_pretrained(checkpoint1)
model_1 = AutoModelForSequenceClassification.from_pretrained(checkpoint1)

# Define the sentences
sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

# Manually tokenize the sentences and convert them to IDs
token_ids = []
for sequence in sequences:
    tokens = tokenizer1.tokenize(sequence)
    ids = tokenizer1.convert_tokens_to_ids(tokens)
    token_ids.append(ids)

# Determine the maximum length of the tokenized sequences
max_len = max(len(ids) for ids in token_ids)

# Pad the sequences to the maximum length
padded_token_ids = []
for ids in token_ids:
    padded_ids = ids + [tokenizer1.pad_token_id] * (max_len - len(ids))
    padded_token_ids.append(padded_ids)

# Convert the padded token IDs to a tensor
input_ids = torch.tensor(padded_token_ids)

# Create attention masks to handle padding
attention_mask = torch.where(input_ids != tokenizer1.pad_token_id, 1, 0)

# Pass the tokenized sequences through the model
outputs = model_1(input_ids, attention_mask=attention_mask)

# Extract logits
logits = outputs.logits

print("Logits:", logits)


