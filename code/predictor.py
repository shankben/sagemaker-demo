import sys
import json
import logging

LOGGER_FORMAT = "---- %(asctime)s %(module)s(%(lineno)d) - %(levelname)s - %(message)s"
logging.basicConfig(stream = sys.stdout, format = LOGGER_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

JSON_CONTENT_TYPE = "application/json"
PRE_TRAINED_MODEL_NAME = "distilbert-base-uncased"
CLASS_NAMES = ["NEGATIVE", "POSITIVE"]

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_path):
        super(SentimentClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(f"Loaded {self.model}")

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        logger.info(f"{output}")
        values, indices = torch.max(output.logits, dim = 1)
        normalized = torch.softmax(output.logits, dim = 1)
        index = indices.item()
        return (normalized[0][index].item(), index)


def model_fn(model_dir):
    print(f"Attempting to load {model_dir}")
    return SentimentClassifier(model_dir)

def predict_fn(input_data, model):
    tokenized = tokenizer(
        input_data["text"],
        add_special_tokens = True,
        return_token_type_ids = False,
        return_attention_mask = True,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )
    confidence, index = model(tokenized["input_ids"], tokenized["attention_mask"])
    logger.info(f"{input_data['text']} -> {CLASS_NAMES[index]} {confidence}")
    return {
        "sentiment": CLASS_NAMES[index],
        "confidence": confidence
    }

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):  
    if content_type != JSON_CONTENT_TYPE:
        pass
    input_data = json.loads(serialized_input_data)
    return input_data

def output_fn(prediction_output, accept = JSON_CONTENT_TYPE):
    if accept != JSON_CONTENT_TYPE:
        raise Exception(f"Requested unsupported ContentType in Accept: {accept}")
    return json.dumps(prediction_output), accept
