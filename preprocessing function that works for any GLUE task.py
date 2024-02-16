from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

def preprocess_glue(dataset_name, task_name, checkpoint, max_length=None, batch_size=32):
    # Load the specified GLUE dataset.
    raw_datasets = load_dataset(dataset_name, task_name)
    
    # Load the tokenizer associated with the specified checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Define a function to tokenize the sentences in the dataset.
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True, max_length=max_length)

    # Apply the tokenize function to all examples in the dataset.
    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

    # Create a data collator.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Get a subset of the training set for demonstration (remove this for actual use).
    samples = tokenized_dataset["train"][:batch_size]

    # Remove the 'idx' and 'sentence' fields from the samples.
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}

    # Print the length of the 'input_ids' field for each sample.
    print("Length of input_ids for each sample:")
    print([len(x) for x in samples["input_ids"]])

    # Form a batch from the samples using the data collator.
    batch = data_collator(samples)

    # Print the shape of each field in the batch.
    print("Shape of each field in the batch:")
    print({k: v.shape for k, v in batch.items()})


# Example usage:
dataset_name = "glue"
task_name = "sst2"
checkpoint = "bert-base-uncased"
max_length = 128
batch_size = 8

preprocess_glue(dataset_name, task_name, checkpoint, max_length=max_length, batch_size=batch_size)
