from datasets import load_dataset
from transformers import AutoTokenizer

#load the MRPC (Microsoft Research Paraphrase Corpus) dataset.
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

#Obtain the train dataset and access the first element
raw_train_datasets = raw_datasets["train"]
raw_train_datasets[0]


#Obtain the test dataset and access the twelfth element
raw_test_datasets = raw_datasets["test"]
raw_test_datasets[11]

#obtain the validation dataset 
raw_validation_datasets = raw_datasets["validation"]


#check the column names and the features
raw_train_datasets.features

#look at element 15 of the train dataset and element 87 of the validation set
print(raw_train_datasets[14])
print(raw_validation_datasets[86])

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentence_1 = tokenizer(raw_train_datasets['sentence1'], padding=True, truncation=True)
tokenized_sentence_2 = tokenizer(raw_train_datasets['sentence2'], padding=True, truncation=True)

#Extract the 15th element of the train dataset and tokenize them as a pair and separately
train_sentence1 = raw_train_datasets[14]['sentence1']
train_sentence2 = raw_train_datasets[14]['sentence2']

tokenize_sentence_1 = tokenizer(train_sentence1, padding=True, truncation=True)
tokenize_sentence_2 = tokenizer(train_sentence2, padding=True, truncation=True)
tokenize_sentence_1_2 = tokenizer(train_sentence1, train_sentence2, padding=True, truncation=True)

print(tokenize_sentence_1)
print(tokenize_sentence_2)
print(tokenize_sentence_1_2)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

tokenized_dataset

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_dataset["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

 

