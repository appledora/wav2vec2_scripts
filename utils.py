import json
import os
import logging
from sox import Transformer
import torch
import gc


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


def extract_all_chars(batch):
    all_text = " ".join(batch["transcripts"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def get_vocab(train_dataset, valid_dataset, BASE_PATH):
    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train_dataset.column_names["train"],
    )
    vocab_valid = valid_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=valid_dataset.column_names["test"],
    )

    vocab_list = list(set(vocab_valid["test"]["vocab"][0]))

    vocab_list = list(set(vocab_train["train"]["vocab"][0]) | set(vocab_valid["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(os.path.join(BASE_PATH, "vocab.json"), "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)


def resample_audio(sampling_rate: int = 16000, num_channel: int = 1, input_file_path: str = "", output_dir: str = ""):
    # audio filename from input_file_path
    audio_filename = os.path.basename(input_file_path)
    # output wavfile path
    output_wav_path = os.path.join(output_dir, audio_filename)
    if not os.path.exists(output_wav_path):
        tfm = Transformer()
        tfm.rate(samplerate=sampling_rate)
        tfm.channels(n_channels=num_channel)
        tfm.build(input_filepath=input_file_path, output_filepath=output_wav_path)

        logging.info(f"Audio sampling done: {output_wav_path}")


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
