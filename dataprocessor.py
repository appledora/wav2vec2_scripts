import regex as re
import emoji
from ftfy import fix_text
import unicodedata
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union

CHAR_REPLACEMENTS = str.maketrans(
    {
        "¹": "1",
        "²": "2",
        "³": "3",
        "À": "A",
        "Á": "A",
        "Â": "A",
        "Ã": "A",
        "Ä": "A",
        "Å": "A",
        "Ā": "A",
        "Ă": "A",
        "Ą": "A",
        "Ǎ": "A",
        "Ǟ": "A",
        "Ǡ": "A",
        "Ǻ": "A",
        "Ȁ": "A",
        "Ȃ": "A",
        "Ȧ": "A",
        "Ḁ": "A",
        "Ạ": "A",
        "Ả": "A",
        "Ấ": "A",
        "Ầ": "A",
        "Ẩ": "A",
        "Ẫ": "A",
        "Ậ": "A",
        "Ắ": "A",
        "Ằ": "A",
        "Ẳ": "A",
        "Ẵ": "A",
        "Ặ": "A",
        "à": "a",
        "á": "a",
        "â": "a",
        "ã": "a",
        "ä": "a",
        "å": "a",
        "ª": "a",
        "ā": "a",
        "ă": "a",
        "ą": "a",
        "ǎ": "a",
        "ǟ": "a",
        "ǡ": "a",
        "ǻ": "a",
        "ȁ": "a",
        "ȃ": "a",
        "ȧ": "a",
        "ḁ": "a",
        "ạ": "a",
        "ả": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "Ḃ": "B",
        "Ḅ": "B",
        "Ḇ": "B",
        "ḃ": "b",
        "ḅ": "b",
        "ḇ": "b",
        "Ç": "C",
        "Ć": "C",
        "Ĉ": "C",
        "Ċ": "C",
        "Č": "C",
        "Ḉ": "C",
        "ç": "c",
        "ć": "c",
        "ĉ": "c",
        "ċ": "c",
        "č": "c",
        "ḉ": "c",
        "Ð": "D",
        "Ď": "D",
        "Đ": "D",
        "Ḋ": "D",
        "Ḍ": "D",
        "Ḏ": "D",
        "Ḑ": "D",
        "Ḓ": "D",
        "ď": "d",
        "đ": "d",
        "ḋ": "d",
        "ḍ": "d",
        "ḏ": "d",
        "ḑ": "d",
        "ḓ": "d",
        "È": "E",
        "É": "E",
        "Ê": "E",
        "Ë": "E",
        "Ē": "E",
        "Ĕ": "E",
        "Ė": "E",
        "Ę": "E",
        "Ě": "E",
        "Ȅ": "E",
        "Ȇ": "E",
        "Ȩ": "E",
        "Ḕ": "E",
        "Ḗ": "E",
        "Ḙ": "E",
        "Ḛ": "E",
        "Ḝ": "E",
        "Ẹ": "E",
        "Ẻ": "E",
        "Ẽ": "E",
        "Ế": "E",
        "Ề": "E",
        "Ể": "E",
        "Ễ": "E",
        "Ệ": "E",
        "è": "e",
        "é": "e",
        "ê": "e",
        "ë": "e",
        "ē": "e",
        "ĕ": "e",
        "ė": "e",
        "ę": "e",
        "ě": "e",
        "ȅ": "e",
        "ȇ": "e",
        "ȩ": "e",
        "ḕ": "e",
        "ḗ": "e",
        "ḙ": "e",
        "ḛ": "e",
        "ḝ": "e",
        "ẹ": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "Ḟ": "F",
        "ḟ": "f",
        "Ĝ": "G",
        "Ğ": "G",
        "Ġ": "G",
        "Ģ": "G",
        "Ǧ": "G",
        "Ǵ": "G",
        "Ḡ": "G",
        "ĝ": "g",
        "ğ": "g",
        "ġ": "g",
        "ģ": "g",
        "ǧ": "g",
        "ǵ": "g",
        "ḡ": "g",
        "Ĥ": "H",
        "Ħ": "H",
        "Ȟ": "H",
        "Ḣ": "H",
        "Ḥ": "H",
        "Ḧ": "H",
        "Ḩ": "H",
        "Ḫ": "H",
        "ĥ": "h",
        "ħ": "h",
        "ȟ": "h",
        "ḣ": "h",
        "ḥ": "h",
        "ḧ": "h",
        "ḩ": "h",
        "ḫ": "h",
        "ẖ": "h",
        "Ì": "I",
        "Í": "I",
        "Î": "I",
        "Ï": "I",
        "Ĩ": "I",
        "Ī": "I",
        "Ĭ": "I",
        "Į": "I",
        "İ": "I",
        "Ǐ": "I",
        "Ȉ": "I",
        "Ȋ": "I",
        "Ḭ": "I",
        "Ḯ": "I",
        "Ỉ": "I",
        "Ị": "I",
        "ì": "i",
        "í": "i",
        "î": "i",
        "ï": "i",
        "ĩ": "i",
        "ī": "i",
        "ĭ": "i",
        "į": "i",
        "ı": "i",
        "ǐ": "i",
        "ȉ": "i",
        "ȋ": "i",
        "ḭ": "i",
        "ḯ": "i",
        "ỉ": "i",
        "ị": "i",
        "Ĵ": "J",
        "ĵ": "j",
        "Ķ": "K",
        "Ǩ": "K",
        "Ḱ": "K",
        "Ḳ": "K",
        "Ḵ": "K",
        "ķ": "k",
        "ǩ": "k",
        "ḱ": "k",
        "ḳ": "k",
        "ḵ": "k",
        "Ĺ": "L",
        "Ļ": "L",
        "Ľ": "L",
        "Ŀ": "L",
        "Ł": "L",
        "Ḷ": "L",
        "Ḹ": "L",
        "Ḻ": "L",
        "Ḽ": "L",
        "ĺ": "l",
        "ļ": "l",
        "ľ": "l",
        "ŀ": "l",
        "ł": "l",
        "ḷ": "l",
        "ḹ": "l",
        "ḻ": "l",
        "ḽ": "l",
        "Ḿ": "M",
        "Ṁ": "M",
        "Ṃ": "M",
        "ḿ": "m",
        "ṁ": "m",
        "ṃ": "m",
        "Ñ": "N",
        "Ń": "N",
        "Ņ": "N",
        "Ň": "N",
        "Ǹ": "N",
        "Ṅ": "N",
        "Ṇ": "N",
        "Ṉ": "N",
        "Ṋ": "N",
        "ñ": "n",
        "ń": "n",
        "ņ": "n",
        "ň": "n",
        "ǹ": "n",
        "ṅ": "n",
        "ṇ": "n",
        "ṉ": "n",
        "ṋ": "n",
        "Ò": "O",
        "Ó": "O",
        "Ô": "O",
        "Õ": "O",
        "Ö": "O",
        "Ō": "O",
        "Ŏ": "O",
        "Ő": "O",
        "Ơ": "O",
        "Ǒ": "O",
        "Ǫ": "O",
        "Ǭ": "O",
        "Ȍ": "O",
        "Ȏ": "O",
        "Ȫ": "O",
        "Ȭ": "O",
        "Ȯ": "O",
        "Ȱ": "O",
        "Ṍ": "O",
        "Ṏ": "O",
        "Ṑ": "O",
        "Ṓ": "O",
        "Ọ": "O",
        "Ỏ": "O",
        "Ố": "O",
        "Ồ": "O",
        "Ổ": "O",
        "Ỗ": "O",
        "Ộ": "O",
        "Ớ": "O",
        "Ờ": "O",
        "Ở": "O",
        "Ỡ": "O",
        "Ợ": "O",
        "ò": "o",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ö": "o",
        "ō": "o",
        "ŏ": "o",
        "ő": "o",
        "ơ": "o",
        "ǒ": "o",
        "ǫ": "o",
        "ǭ": "o",
        "ȍ": "o",
        "ȏ": "o",
        "ȫ": "o",
        "ȭ": "o",
        "ȯ": "o",
        "ȱ": "o",
        "ṍ": "o",
        "ṏ": "o",
        "ṑ": "o",
        "ṓ": "o",
        "ọ": "o",
        "ỏ": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "Ṕ": "P",
        "Ṗ": "P",
        "ṕ": "p",
        "ṗ": "p",
        "Ŕ": "R",
        "Ŗ": "R",
        "Ř": "R",
        "Ȑ": "R",
        "Ȓ": "R",
        "Ṙ": "R",
        "Ṛ": "R",
        "Ṝ": "R",
        "Ṟ": "R",
        "ŕ": "r",
        "ŗ": "r",
        "ř": "r",
        "ȑ": "r",
        "ȓ": "r",
        "ṙ": "r",
        "ṛ": "r",
        "ṝ": "r",
        "ṟ": "r",
        "Ś": "S",
        "Ŝ": "S",
        "Ş": "S",
        "Š": "s",
        "Ș": "S",
        "Ṡ": "S",
        "Ṣ": "S",
        "Ṥ": "S",
        "Ṧ": "S",
        "Ṩ": "S",
        "ś": "s",
        "ŝ": "s",
        "ş": "s",
        "š": "s",
        "ș": "s",
        "ṡ": "s",
        "ṣ": "s",
        "ṥ": "s",
        "ṧ": "s",
        "ṩ": "s",
        "Ţ": "T",
        "Ť": "T",
        "Ŧ": "T",
        "Ț": "T",
        "Ṫ": "T",
        "Ṭ": "T",
        "Ṯ": "T",
        "Ṱ": "T",
        "ţ": "t",
        "ť": "t",
        "ŧ": "t",
        "ț": "t",
        "ṫ": "t",
        "ṭ": "t",
        "ṯ": "t",
        "ṱ": "t",
        "ẗ": "t",
        "Ù": "U",
        "Ú": "U",
        "Û": "U",
        "Ü": "U",
        "Ũ": "U",
        "Ū": "U",
        "Ŭ": "U",
        "Ů": "U",
        "Ű": "U",
        "Ų": "U",
        "Ư": "U",
        "Ǔ": "U",
        "Ǖ": "U",
        "Ǘ": "U",
        "Ǚ": "U",
        "Ǜ": "U",
        "Ȕ": "U",
        "Ȗ": "U",
        "Ṳ": "U",
        "Ṵ": "U",
        "Ṷ": "U",
        "Ṹ": "U",
        "Ṻ": "U",
        "Ụ": "U",
        "Ủ": "U",
        "Ứ": "U",
        "Ừ": "U",
        "Ử": "U",
        "Ữ": "U",
        "Ự": "U",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "u",
        "ũ": "u",
        "ū": "u",
        "ŭ": "u",
        "ů": "u",
        "ű": "u",
        "ų": "u",
        "ư": "u",
        "ǔ": "u",
        "ǖ": "u",
        "ǘ": "u",
        "ǚ": "u",
        "ǜ": "u",
        "ȕ": "u",
        "ȗ": "u",
        "ṳ": "u",
        "ṵ": "u",
        "ṷ": "u",
        "ṹ": "u",
        "ṻ": "u",
        "ụ": "u",
        "ủ": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "Ṽ": "V",
        "Ṿ": "V",
        "ṽ": "v",
        "ṿ": "v",
        "Ŵ": "W",
        "Ẁ": "W",
        "Ẃ": "W",
        "Ẅ": "W",
        "Ẇ": "W",
        "Ẉ": "W",
        "ŵ": "w",
        "ẁ": "w",
        "ẃ": "w",
        "ẅ": "w",
        "ẇ": "w",
        "ẉ": "w",
        "ẘ": "w",
        "Ẋ": "X",
        "Ẍ": "X",
        "ẋ": "x",
        "ẍ": "x",
        "Ý": "y",
        "Ŷ": "Y",
        "Ÿ": "Y",
        "Ȳ": "Y",
        "Ẏ": "Y",
        "Ỳ": "Y",
        "Ỵ": "Y",
        "Ỷ": "Y",
        "Ỹ": "Y",
        "ý": "y",
        "ÿ": "y",
        "ŷ": "y",
        "ȳ": "y",
        "ẏ": "y",
        "ỳ": "y",
        "ỵ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ẙ": "y",
        "Ź": "Z",
        "Ż": "Z",
        "Ž": "Z",
        "Ẑ": "Z",
        "Ẓ": "Z",
        "Ẕ": "Z",
        "ź": "z",
        "ż": "z",
        "ž": "z",
        "ẑ": "z",
        "ẓ": "z",
        "ẕ": "z",
        "Ĳ": "IJ",
        "ĳ": "ij",
        "ø": "o",
        "Ø": "O",
        "ɨ": "i",
        "ð": "d",
    }
)

UNICODE_REPLACEMENTS = {
    "\u00ad": "",
    "\u09af\u09bc": "\u09df",
    "\u09a2\u09bc": "\u09dd",
    "\u09a1\u09bc": "\u09dc",
    "\u09ac\u09bc": "\u09b0",
    "\u09c7\u09be": "\u09cb",
    "\u09c7\u09d7": "\u09cc",
    "\u0985\u09be": "\u0986",
    "\u09c7\u0981\u09d7": "\u09cc\u0981",
    "\u09c7\u0981\u09be": "\u09cb\u0981",
    "\u09c7([^\u09d7])\u09d7": "\\g<1>\u09cc",
    "\u00a0": " ",
    "\u200d": "",
    "\u200b": "",
    "\u2060": "",
    "\u201e": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": " - ",
    "\u00b4": "'",
    "\u2018": "'",
    "\u201a": "'",
    "\u2019": "'",
    "\u00b4\u00b4": '"',
    "\u2026": "...",
    "\u00a0\u00ab\u00a0": '"',
    "\u00ab\u00a0": '"',
    "\u00ab": '"',
    "\u00a0\u00bb\u00a0": '"',
    "\u00a0\u00bb": '"',
    "\u00bb": '"',
    "\u09f7": "\u0964",
    "\uff0c": ",",
    "\u3001": ",",
    "\u2236": ":",
    "\uff1a": ":",
    "\uff1f": "?",
    "\u300a": '"',
    "\u300b": '"',
    "\uff09": ")",
    "\uff01": "!",
    "\uff08": "(",
    "\uff1b": ";",
    "\u300d": '"',
    "\u300c": '"',
    "\uff10": "0",
    "\uff11": "1",
    "\uff12": "2",
    "\uff13": "3",
    "\uff14": "4",
    "\uff15": "5",
    "\uff16": "6",
    "\uff17": "7",
    "\uff18": "8",
    "\uff19": "9",
    "\uff5e": "~",
    "\u2501": "-",
    "\u3008": "<",
    "\u3009": ">",
    "\u3010": "[",
    "\u3011": "]",
    "\uff05": "%",
}

UNICODE_REPLACEMENTS_REGEX = re.compile("|".join(UNICODE_REPLACEMENTS.keys()))

DOUBLE_QUOTE_REGEX = re.compile(
    "|".join(
        [
            "«",
            "‹",
            "»",
            "›",
            "„",
            "“",
            "‟",
            "”",
            "❝",
            "❞",
            "❮",
            "❯",
            "〝",
            "〞",
            "〟",
            "＂",
        ]
    )
)

SINGLE_QUOTE_REGEX = re.compile("|".join(["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]))

URL_HANDLER_REGEX = re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    # r"(?:$|(?![\w?!+&\/\)]))",
    # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
    # But I made sure that it does not include ), ] and } in the URL.
    flags=re.UNICODE | re.IGNORECASE,
)

WHITESPACE_HANDLER_REGEX = re.compile(r"\s+")
PUNCT_HANDLER_REGEX = re.compile(r"\p{punct}")
EMOJI_HANDLER_REGEX = emoji.get_emoji_regexp()


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
        lambda match: UNICODE_REPLACEMENTS.get(match.group(0), f"{match.group(1)}\u09cc"), text
    )

    if apply_unicode_norm_last:
        text = unicodedata.normalize(unicode_norm, text)

    # finally clean up extra whitespaces
    text = WHITESPACE_HANDLER_REGEX.sub(" ", text)
    row["transcripts"] = text
    return row


def prepare_dataset(batch, processor=processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcripts"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Filter out any features that have empty inputs or labels
        valid_features = [
            feature for feature in features if len(feature["input_values"]) > 0 and len(feature["labels"]) > 0
        ]

        if not valid_features:
            # Handle the case where all features are empty (no valid data) by returning a minimal batch
            return self._create_empty_batch()

        # Split inputs and labels for valid features
        input_features = [{"input_values": feature["input_values"]} for feature in valid_features]
        label_features = [{"input_ids": feature["labels"]} for feature in valid_features]

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
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

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
        empty_batch = {"input_values": input_ids, "attention_mask": attention_mask, "labels": labels}

        # print(empty_batch)
        return empty_batch
