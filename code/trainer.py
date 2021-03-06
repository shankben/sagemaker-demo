from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    ## Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level = logging.getLevelName("INFO"),
        handlers = [logging.StreamHandler(sys.stdout)],
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ## Load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = "binary")
        acc = accuracy_score(labels, preds)

#         print(f"'eval_accuracy': {acc}")
#         print(f"'eval_f1': {f1}")
#         print(f"'eval_precision': {precision}")
#         print(f"'eval_recall': {recall}")

        return {
            "accuracy": acc, 
            "f1": f1, 
            "precision": precision, 
            "recall": recall
        }

    trainer = Trainer(
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name),
        compute_metrics = compute_metrics,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        args = TrainingArguments(
            output_dir = args.model_dir,
            num_train_epochs = args.epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            warmup_steps = args.warmup_steps,
            evaluation_strategy = "epoch",
            logging_dir = f"{args.output_data_dir}/logs",
            logging_steps = args.logging_steps,
            learning_rate = float(args.learning_rate)
        )
    )

    trainer.train()

    eval_result = trainer.evaluate(eval_dataset = test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    trainer.save_model(args.model_dir)

