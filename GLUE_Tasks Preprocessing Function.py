from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# Loading the GLUE dataset.
task_name = "mrpc"  # Change this to the specific task you want to preprocess
raw_datasets = load_dataset("glue", task_name)

# Selecting a checkpoint and tokenizer.
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define the preprocessing function.
def preprocess_function(example):
    # Check if the task requires two sentences or one sentence
    if 'sentence2' in example:
        # For tasks with two sentences
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding='max_length', max_length=128)
    else:
        # For tasks with only one sentence
        return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=128)

# Tokenizing the dataset.
tokenized_dataset = raw_datasets.map(preprocess_function, batched=True)

# Creating a data collator.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
