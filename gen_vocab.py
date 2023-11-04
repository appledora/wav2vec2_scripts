import json
import os
def extract_all_chars(batch):
  all_text = " ".join(batch["transcripts"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


def get_vocab(train_dataset, test_dataset, valid_dataset, BASE_PATH):

    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names["train"])
    vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_dataset.column_names["test"])
    vocab_valid = valid_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=valid_dataset.column_names["test"])

    vocab_list = list(set(vocab_valid["test"]["vocab"][0]))

    vocab_list = list(set(vocab_train["train"]["vocab"][0]) | set(vocab_test["test"]["vocab"][0]) | set(vocab_valid["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(os.path.join(BASE_PATH, 'vocab.json'), 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)