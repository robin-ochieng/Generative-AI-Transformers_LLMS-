# Importing the necessary libraries
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
  
# Loading the GLUE SST-2 dataset, checkpoint and tokenizer.
raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to tokenize the sentences in the dataset.
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

# Applying the tokenize function to all examples in the dataset.
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Creating a data collator for padding.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Defining the training arguments for customizing the training process.
training_args = TrainingArguments("test-trainer")

# Defining the model with the checkpoint and the number of labels. 
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Creating a Trainer instance. This will be used to train the model, and contains the data(training and validation set), model, and training arguments.
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,)

# Training the model
trainer.train()

# Making predictions on the validation set
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# Getting the index of the maximum value in predictions
preds = np.argmax(predictions.predictions, axis=-1)

# Loading the evaluation metric
metric = load_metric("glue", "sst2")
# Computing the metric using the predictions and the actual labels
metric.compute(predictions=preds, references=predictions.label_ids)

# Function to compute metrics for evaluation predictions
def compute_metrics(eval_preds):
    metric = load_metric("glue", "sst2")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Redefining the training arguments to include evaluation strategy
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
# Redefining the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Creating a new Trainer instance with the compute_metrics function
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Training the model again, this time with evaluation at each epoch
trainer.train()