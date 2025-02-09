import os
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM

#Switch between train and pretrained models
USE_PRETRAINED = True

if USE_PRETRAINED:
    # Pretrained model settings
    TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    MODEL = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
else:
    # Locally trained model settings
    TOKENIZER = DistilBertTokenizerFast.from_pretrained('model')
    MODEL = DistilBertForMaskedLM.from_pretrained('model')
