import torch
import regex as re
from ftfy import fix_text
import unicodedata
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union
from const import (
    SINGLE_QUOTE_REGEX,
    DOUBLE_QUOTE_REGEX,
    PUNCT_HANDLER_REGEX,
    URL_HANDLER_REGEX,
    EMOJI_HANDLER_REGEX,
    CHAR_REPLACEMENTS,
    UNICODE_REPLACEMENTS_REGEX,
    UNICODE_REPLACEMENTS,
    WHITESPACE_HANDLER_REGEX,
)


def fix_quotes(text):
    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def remove_english_characters(string):
    pattern = re.compile("[0-9a-zA-Z<>]")
    return pattern.sub("", string)


def normalize(
    row,
    unicode_norm="NFKC",
    punct_replacement=None,
    url_replacement=None,
    emoji_replacement=None,
    apply_unicode_norm_last=True,
):
    # fix encoding related issues first
    # and group characters for future
    # char replacements to work
    text = row["transcripts"]
    text = fix_text(
        text,
        normalization="NFC",
        explain=False,
    )

    # normalize variations of quotes
    text = fix_quotes(text)
    text = remove_english_characters(text)
    # replace punctuations with specified replacement (if any)
    if punct_replacement is not None:
        text = PUNCT_HANDLER_REGEX.sub(punct_replacement, text)

    # replace URLS in text with specified replacement (if any)
    if url_replacement is not None:
        text = URL_HANDLER_REGEX.sub(url_replacement, text)

    # replace emojis in text with specified replacement (if any)
    if emoji_replacement is not None:
        text = EMOJI_HANDLER_REGEX.sub(emoji_replacement, text)

    # apply char replacements
    text = text.translate(CHAR_REPLACEMENTS)

    if not apply_unicode_norm_last:
        text = unicodedata.normalize(text, unicode_norm)

    # apply unicode replacements
    text = UNICODE_REPLACEMENTS_REGEX.sub(
        lambda match: UNICODE_REPLACEMENTS.get(
            match.group(0), f"{match.group(1)}\u09cc"
        ),
        text,
    )

    if apply_unicode_norm_last:
        text = unicodedata.normalize(unicode_norm, text)

    # finally clean up extra whitespaces
    text = WHITESPACE_HANDLER_REGEX.sub(" ", text)
    row["transcripts"] = text
    return row


def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcripts"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Filter out any features that have empty inputs or labels
        valid_features = [
            feature
            for feature in features
            if len(feature["input_values"]) > 0 and len(feature["labels"]) > 0
        ]

        if not valid_features:
            # Handle the case where all features are empty (no valid data) by returning a minimal batch
            return self._create_empty_batch()

        # Split inputs and labels for valid features
        input_features = [
            {"input_values": feature["input_values"]} for feature in valid_features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in valid_features
        ]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        # print(len(batch["input_values"][0]), "\t", len(batch["attention_mask"][0]), "\t", len(batch["labels"][0]))
        # print("batch: ", batch)
        return batch

    def _create_empty_batch(self):
        # Get the model's expected input size

        # Create an empty batch with attention_mask to prevent training on it
        input_ids = torch.zeros((1, 248832), dtype=torch.float)
        attention_mask = torch.ones((1, 248832), dtype=torch.int32)
        labels = torch.zeros((1, 71), dtype=torch.long)
        empty_batch = {
            "input_values": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # print(empty_batch)
        return empty_batch
