from dataprocessor import normalize, DataCollatorCTCWithPadding
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
import gc
import os
import torch
import numpy
import pandas as pd


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


BASE_PATH = "AUDIO_DIR"
TRAIN_DIR = "TRAIN_FILES_DIR"
VALID_DIR = "VALID_FILES_DIR"

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


tokenizer = Wav2Vec2CTCTokenizer(
    os.path.join(BASE_PATH, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)


feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=False, return_attention_mask=True
)


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


clean_memory()

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
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
model = model.to("cuda")
model.config.ctc_zero_infinity = True
model.freeze_feature_encoder()


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcripts"]).input_ids
    return batch


train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names["train"])
train_dataset = train_dataset.filter(lambda example: example is not None)


valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=valid_dataset.column_names["test"])
valid_dataset = valid_dataset.filter(lambda example: example is not None)


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


wer_metric = load_metric("wer")
cer_metric = load_metric("cer", revision="master")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


training_args = TrainingArguments(
    output_dir="OUTPUT_DIR",
    group_by_length=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.005,
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    gradient_accumulation_steps=10,
    evaluation_strategy="steps",
    num_train_epochs=100,
    gradient_checkpointing=True,
    fp16=False,
    save_steps=500,
    eval_steps=500,
    logging_steps=400,
    warmup_steps=500,
    save_total_limit=5,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    log_level="debug",
)

clean_memory()


torch.cuda.empty_cache()
gc.collect()
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset["train"],
    eval_dataset=valid_dataset["test"],
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0001)],
)
trainer.train(resume_from_checkpoint=False)
trainer.save_model("w2v_all_cer")
loss_df = pd.DataFrame(trainer.state.log_history)
loss_df.to_csv("loss.csv", index=False)
