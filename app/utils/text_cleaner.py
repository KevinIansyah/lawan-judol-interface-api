import re
from unidecode import unidecode

def clean_text(text):
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
