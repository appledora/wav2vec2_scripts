import argparse
import pandas as pd
import os
import torch
from dataprocessor import normalize, prepare_dataset
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def init_model(REPO_NAME):
    model = AutoModelForCTC.from_pretrained(REPO_NAME)
    model.to(device)
    print(f"Model loaded: {REPO_NAME}")


def map_to_result(batch, processor):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    # print(">: ", batch["pred_str"])
    # print(">>: ", batch["text"])
    return batch


def __main__():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--audio_dir",
        type=str,
        default="AUDIO_DIR",
        help="Base directory containing split folders",
    )
    args.add_argument(
        "--test_dir", type=str, default="TEST_FILES_DIR", help="Test data directory"
    )
    args.add_argument(
        "--ckpt_dir",
        type=str,
        default="SAVED_MODEL_PATH",
        help="Fine-tuned model checkpoint",
    )
    args.add_argument(
        "--result_dir",
        type=str,
        default="RESULT_DIR",
        help="Location to save the inference results",
    )

    args = args.parse_args()
    BASE_PATH = args.audio_dir
    TEST_DIR = args.test_dir
    TEST_PATH = os.path.join(BASE_PATH, TEST_DIR)
    REPO_NAME = args.ckpt_dir
    RESULT_DIR = args.result_dir

    test_dataset = load_dataset("audiofolder", data_dir=TEST_PATH, name="test_audio")
    test_dataset = test_dataset.map(normalize)
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
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    test_dataset = test_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=test_dataset.column_names["test"],
    )
    test_dataset = test_dataset.filter(lambda example: example is not None)
    init_model(REPO_NAME)
    results = test_dataset["test"].map(
        map_to_result, remove_columns=test_dataset["test"].column_names
    )
    results = results.filter(
        lambda result: len(result["pred_str"]) > 0 and len(result["text"]) > 0
    )

    wer_metric = load_metric("wer")
    print(
        "Test WER: {:.3f}".format(
            wer_metric.compute(
                predictions=results["pred_str"], references=results["text"]
            )
        )
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "results.csv"))
    print("Inference results saved to: ", RESULT_DIR)


if __name__ == "__main__":
    __main__()
