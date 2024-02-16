# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

# Load the GLUE SST-2 dataset.
# GLUE (General Language Understanding Evaluation) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
raw_datasets = load_dataset("glue", "sst2")
print(raw_datasets)  # Print the loaded dataset

# Define the checkpoint as 'bert-base-uncased'. This is a pre-trained model provided by Hugging Face.
checkpoint = "bert-base-uncased"

# Load the tokenizer associated with the 'bert-base-uncased' model.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define a function to tokenize the sentences in the dataset.
# The tokenizer converts the sentences into a format that the model can understand.
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

# Apply the tokenize function to all examples in the dataset.
tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

# Create a data collator.
# A data collator is used to form a batch by padding all the sentences to the same length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Select the first 8 samples from the training set of the tokenized dataset.
samples = tokenized_dataset["train"][:8]

# Remove the 'idx' and 'sentence' fields from the samples.
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}

# Print the length of the 'input_ids' field for each sample.
# 'input_ids' are the tokenized representation of the sentences.
[len(x) for x in samples["input_ids"]]

# Form a batch from the samples using the data collator.
batch = data_collator(samples)

# Print the shape of each field in the batch.
{k: v.shape for k, v in batch.items()}