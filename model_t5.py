import os
import sys
import argparse
import tensorflow as tf
from sklearn.metrics import f1_score
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, \
    AdamWeightDecay, AutoConfig, TFAutoModelForSeq2SeqLM, TFT5ForConditionalGeneration,\
    T5Tokenizer, T5ForConditionalGeneration


def parse_args():
    """ Perform command-line argument parsing. """
    
    parser = argparse.ArgumentParser(
        description="Fine-tuning Summarization Model (T5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mode',
        required=True,
        choices=['train', 'load'],
        help='''Select whether to train or load the model specified in model path.''')
    parser.add_argument(
        '--load-t5',
        default='t5_small_news',
        help='''Enter the path to yiyr fine-trained T5 model.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model.''')
    parser.add_argument(
        '--save-t5',
        default='t5_small_news',
        help='''Enter the path to save your trained model.''')

    return parser.parse_args()


def preprocess_function(examples):
    """ Use tokenizer to preprocess data. """
    
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    prefix = "summarize: "

    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=80, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def download_and_preprocess_data():
    """ Load dataset from HuggingFace and preprocess. """
    
    news_ds = load_dataset("cnn_dailymail", "3.0.0")

    # Tokenized using preprocess_function
    tokenized_news = news_ds.map(preprocess_function, batched=True)

    return tokenized_news


def train_val_test_split(ds, data_collator):
    """ Split a dataset into train/val/test dataset. """
    
    train_ds = ds["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=True,
        batch_size=2,
        collate_fn=data_collator,
    )

    val_ds = ds["validation"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=False,
        batch_size=2,
        collate_fn=data_collator,
    )

    test_ds = ds["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=False,
        batch_size=1,
        collate_fn=data_collator,
    )

    return train_ds, val_ds, test_ds


def train(model, train_data, val_data):
    """ Training routine. """

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(
        x=train_data, 
        validation_data=val_data, 
        epochs=20,
        shuffle=True,
        steps_per_epoch=8000,
        validation_steps=800,
        # callbacks=[callback],
    )


def test(model, test_data):
    """ Testing routine. """

    model.evaluate(
        x=test_data,
        verbose=1
    )


def generate_summary(model, tokenizer, test_ds):
    """ Use the model to generate summaries. """
    
    for item in test_ds:
        actual = item['labels']
        pred = model.generate(
            input_ids=item['input_ids'],
            max_length=80,
        )

        for i in range(actual.shape[0]):
            pred_sentence = pred[i]
            pred_sentence = tokenizer.decode(pred_sentence, skip_special_tokens=True)
            
            actual_sentence = actual[i].numpy()
            actual_sentence = actual_sentence[actual_sentence >= 0]
            actual_sentence = tokenizer.decode(actual_sentence, skip_special_tokens=True)
            
            print(f"pred = {pred_sentence}")
            print(f"actual = {actual_sentence}")


def compute_metrics(model, tokenizer, metric, test_ds):
    """ Compute the model's performance on test_ds on given metrics. """

    for item in test_ds:
        actual = item['labels']
        pred = model.generate(
            input_ids=item['input_ids'], 
            # attention_mask=item['attention_mask']
        )
        metric.add_batch(predictions=pred, references=actual)

    final_score = metric.compute()
    print(final_score)

    return final_score


def simple_accuracy(preds, labels):
    """ Compute Accuracy (EM) score. """
    
    return (preds == labels).mean().item()


def acc_and_f1(preds, labels):
    """ Compute Accuracy (EM) and F1 score. """
    
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds).item()
    
    return {
        "accuracy": acc,
        "f1": f1,
    }


def main():
    """ Main function. """

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenized_news = download_and_preprocess_data()
    
    optimizer = AdamWeightDecay(
        learning_rate=2e-5, 
        weight_decay_rate=0.01
    )
    
    if ARGS.mode == 'train':
        model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

    else:
        model = TFT5ForConditionalGeneration.from_pretrained(ARGS.load_t5)

    model.compile(optimizer=optimizer)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        return_tensors="tf",
    )

    train_ds, val_ds, test_ds = train_val_test_split(tokenized_news, data_collator)

    if ARGS.evaluate:
        metric = load_metric('rouge')
        compute_metrics(model, tokenizer, metric, test_ds)
    else:
        train(model, train_ds, val_ds)
        test(model, test_ds)
        if ARGS.save_t5 is not None:
            if not os.path.exists(ARGS.save_t5):
                os.makedirs(ARGS.save_t5)
            model.save_pretrained(ARGS.save_t5)
        
    # generate_summary(model, tokenizer, test_ds)

# Make arguments global
ARGS = parse_args()

main()
