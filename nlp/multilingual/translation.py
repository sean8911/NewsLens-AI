import re
from collections import Counter

from nlp.multilingual.glossary import validate_terms
from nlp.multilingual.language import detect_language

POLICY = "excerpt-translation-v3-m2m100;original-citations;language-number-place-checks"
INSUFFICIENT = {
    "en": "I don't have enough information in the available articles to answer that question.",
    "ms": "Maklumat dalam artikel yang tersedia tidak mencukupi untuk menjawab soalan itu.",
    "zh": "现有文章中的信息不足以回答这个问题。",
}
INTRO = {
    "en": "The retrieved reports provide these details:",
    "ms": "Laporan yang ditemukan memberikan butiran berikut:",
    "zh": "检索到的报道提供了以下信息：",
}
TRANSLATED = {"en": "Translated excerpt", "ms": "Petikan terjemahan", "zh": "译文"}


def translate_excerpt(text: str, language: str, translator) -> tuple[str, bool]:
    source = detect_language(text).code
    if source == language:
        return text, False
    if translator is None:
        raise RuntimeError("Translation provider unavailable")
    raw = translator.translate(text, source, language)
    if not isinstance(raw, str) or len(raw) > 2000:
        raise ValueError("Invalid translation output")
    translated = raw.strip()
    validate_terms(text, translated, language)
    if (
        not translated
        or detect_language(translated).code != language
        or Counter(re.findall(r"\d+", text)) != Counter(re.findall(r"\d+", translated))
        or re.search(r"https?://|\[\d+\]", translated)
    ):
        raise ValueError("Invalid translation")
    return translated, True
