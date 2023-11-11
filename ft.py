import argparse
import logging
from typing import Any
import torch
from dataprocessor import normalize, DataCollatorCTCWithPadding, prepare_dataset
from utils import compute_metrics, clean_memory, get_vocab
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
import os
import pandas as pd

wer_metric = load_metric("wer")
cer_metric = load_metric("cer", revision="master")
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_wav2vec2(
    train_dataset,
    valid_dataset,
    processor,
    data_collator,
    repo_name="facebook/wav2vec2-xls-r-300m",
):
    # add arguments to run this script:
    parser = HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_json_file(json_file="args.json")
    clean_memory()
    model = Wav2Vec2ForCTC.from_pretrained(
        repo_name,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        gradient_checkpointing=True,
    )
    model = model.to(device)
    logging.info("Model loaded.")
    model.config.ctc_zero_infinity = True
    model.freeze_feature_encoder()

    clean_memory()
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset["train"],
        eval_dataset=valid_dataset["test"],
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0001
            )
        ],
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("w2v_all_cer")
    logging.info("Model saved.")
    loss_df = pd.DataFrame(trainer.state.log_history)
    loss_df.to_csv("loss_history.csv", index=False)

    clean_memory()


def __main__():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--train_dir",
        type=str,
        default="TRAIN_FILES_DIR",
        help="Training data directory",
    )
    args.add_argument(
        "--valid_dir",
        type=str,
        default="VALID_FILES_DIR",
        help="Validation data directory",
    )
    args.add_argument(
        "--audio_dir",
        type=str,
        default="AUDIO_DIR",
        help="Base directory containing split folders",
    )
    args.add_argument(
        "--repo_name",
        type=str,
        default="SAVED_MODEL_PATH",
        help="Pretrained model repo name",
    )
    args.add_argument(
        "--generate_vocab", type=bool, default=False, help="Generate vocab.json file"
    )

    args = args.parse_args()
    BASE_PATH = args.audio_dir
    TRAIN_DIR = args.train_dir
    VALID_DIR = args.valid_dir
    gen_vocab = args.generate_vocab

    TRAIN_PATH = os.path.join(BASE_PATH, TRAIN_DIR)
    VALID_PATH = os.path.join(BASE_PATH, VALID_DIR)

    # Each of the audiofolder should have the following structure:
    # - [SPLIT]_FILES_DIR
    #   - 1.wav
    #   - 2.wav
    #   - ...
    #   - metadata.csv
    train_dataset = load_dataset("audiofolder", data_dir=TRAIN_PATH, name="train_audio")
    valid_dataset = load_dataset("audiofolder", data_dir=VALID_PATH, name="valid_audio")

    train_dataset = train_dataset.map(normalize)
    valid_dataset = valid_dataset.map(normalize)

    if gen_vocab:
        get_vocab(train_dataset, valid_dataset, BASE_PATH)
        logging.info(
            "Vocab file generated at {}".format(os.path.join(BASE_PATH, "vocab.json"))
        )
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(BASE_PATH, "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=False,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor=processor),
        remove_columns=train_dataset.column_names["train"],
    )
    train_dataset = train_dataset.filter(lambda example: example is not None)
    logging.info("Train dataset loaded.")

    valid_dataset = valid_dataset.map(
        lambda x: prepare_dataset(x, processor=processor),
        remove_columns=valid_dataset.column_names["test"],
    )
    valid_dataset = valid_dataset.filter(lambda example: example is not None)
    logging.info("Validation dataset loaded.")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    train_wav2vec2(
        train_dataset,
        valid_dataset,
        repo_name=args.repo_name,
        processor=processor,
        data_collator=data_collator,
    )


if __name__ == "__main__":
    __main__()
