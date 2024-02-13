from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This boy works at [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This girl works at [MASK].")
print([r["token_str"] for r in result])


classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels)


unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")
