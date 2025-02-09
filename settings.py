from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, RobertaTokenizerFast, RobertaForMaskedLM

# Set the model to use
USE_MODEL = "roberta"

# Define model choices
MODEL_CONFIG = {
    "distilbert": {
        "tokenizer": DistilBertTokenizerFast.from_pretrained,
        "model": DistilBertForMaskedLM.from_pretrained,
        "pretrained_name": "distilbert-base-uncased"
    },
    "roberta": {
        "tokenizer": RobertaTokenizerFast.from_pretrained,
        "model": RobertaForMaskedLM.from_pretrained,
        "pretrained_name": "roberta-base"
    },
    "local": {
        "tokenizer": DistilBertTokenizerFast.from_pretrained,
        "model": DistilBertForMaskedLM.from_pretrained,
        "pretrained_name": "model"
    }
}

config = MODEL_CONFIG.get(USE_MODEL, MODEL_CONFIG["local"])
TOKENIZER = config["tokenizer"](config["pretrained_name"])
MODEL = config["model"](config["pretrained_name"])

