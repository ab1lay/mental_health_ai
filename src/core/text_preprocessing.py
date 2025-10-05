import re

def preprocess_text(text: str) -> str:
    """Очищает текст от эмодзи, спецсимволов и лишних пробелов."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()
