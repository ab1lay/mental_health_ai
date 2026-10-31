import re


def preprocess_text(text: str) -> str:
    """Очищает текст от ссылок, лишних символов и приводит к нижнему регистру.

    Апострофы и дефисы сохраняются: без них "can't breathe" превращается в
    "cant breathe", "self-harm" в "selfharm", и сигналы риска перестают находиться.
    """
    text = (text or "").replace("’", "'")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()
