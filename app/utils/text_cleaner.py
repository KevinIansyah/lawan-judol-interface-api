import re
import html
from unidecode import unidecode

def clean_text_classifier(text):
    if not text or not isinstance(text, str):
        return text

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\B@@\w+', '', text)
    text = unidecode(text)
    hashtags = re.findall(r'#\w+', text)
    for i, tag in enumerate(hashtags):
        text = text.replace(tag, f" <HASHTAG_{i}> ")
    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for i, tag in enumerate(hashtags):
        text = text.replace(f"HASHTAG_{i}", tag)

    return text

def clean_text_keywoard(text):
    if not text or not isinstance(text, str):
        return text

    text = html.unescape(text)
    text = re.sub(r'<[^>]*>', ' ', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002B00-\U00002BFF"  # arrows & misc symbols
        "\U00002300-\U000023FF"  # misc technical
        "\U0001F7E0-\U0001F7EB"  # geometric shapes extended
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F100-\U0001F1FF"  # enclosed alphanumeric supplement (🆗🆘🆕🆓🆙🆒 dll)
        "\U0001F200-\U0001F2FF"  # enclosed ideographic supplement (🈁🈂️🈷️ dll)
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text
