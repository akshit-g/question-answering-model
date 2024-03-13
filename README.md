# DistilBERT Question Answering

## Overview

This project demonstrates how to perform extractive question answering using the DistilBERT model. Extractive question answering involves extracting the answer to a question from a given context or passage.

The model used in this project is [DistilBERT](https://huggingface.co/akshit-g/distilbert-base-cased), a smaller and faster version of the BERT model, trained by Hugging Face.

You can try the model interactively on the Hugging Face Spaces platform [here](https://huggingface.co/spaces/akshit-g/akshit-g-distilbert-base-cased).

## Requirements

- Python 3.x
- transformers library from Hugging Face
- datasets library from Hugging Face

Here is an example to use the model:
```
from transformers import pipeline

model_checkpoint = "akshit-g/distilbert-base-cased"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = "DistilBERT is a lighter version of BERT, developed by Hugging Face."
question = "What is DistilBERT?"
question_answerer(question=question, context=context)
```
