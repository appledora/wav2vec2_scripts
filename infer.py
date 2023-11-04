import os
import torch
from dataprocessor import normalize
from datasets import load_dataset, load_metric
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer

BASE_PATH = "AUDIO_DIR"
REPO_NAME = "SAVED_MODEL_PATH"
TEST_DIR = "TEST_FILES_DIR"
TEST_PATH = os.path.join(BASE_PATH, TEST_DIR)
test_dataset = load_dataset("audiofolder", data_dir=TEST_PATH, name="test_audio")
test_dataset = test_dataset.map(normalize)


tokenizer = Wav2Vec2CTCTokenizer(
    os.path.join(BASE_PATH, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
model = AutoModelForCTC.from_pretrained(REPO_NAME)
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcripts"]).input_ids

    return batch


model.to("cuda")


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    print(">: ", batch["pred_str"])
    print(">>: ", batch["text"])
    return batch


test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names["test"])
test_dataset = test_dataset.filter(lambda example: example is not None)

results = test_dataset["test"].map(map_to_result, remove_columns=test_dataset["test"].column_names)
results = results.filter(lambda result: len(result["pred_str"]) > 0 and len(result["text"]) > 0)


wer_metric = load_metric("wer")

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
