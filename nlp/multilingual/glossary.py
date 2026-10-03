"""Reviewed Malaysian place names; prevent transliteration drift in translated news.

Reference: https://ebrochures.malaysia.travel/cn/malaysia-travel-guide/
The glossary is deliberately bounded; it does not claim to recognize every named entity.
"""

import re

# English/Malay canonical spelling, simplified Chinese, traditional Chinese aliases.
PLACES = (
    ("Malaysia", "马来西亚", "馬來西亞"),
    ("Kuala Lumpur", "吉隆坡"),
    ("Putrajaya", "布城"),
    ("Labuan", "纳闽", "納閩"),
    ("Johor", "柔佛"),
    ("Kedah", "吉打"),
    ("Kelantan", "吉兰丹", "吉蘭丹"),
    ("Melaka", "马六甲", "馬六甲", "Malacca"),
    ("Negeri Sembilan", "森美兰", "森美蘭"),
    ("Pahang", "彭亨"),
    ("Penang", "槟城", "檳城", "Pulau Pinang"),
    ("Perak", "霹雳", "霹靂"),
    ("Perlis", "玻璃市"),
    ("Sabah", "沙巴"),
    ("Sarawak", "砂拉越"),
    ("Selangor", "雪兰莪", "雪蘭莪"),
    ("Terengganu", "登嘉楼", "登嘉樓"),
)
ALIASES = {alias: place for place in PLACES for alias in place}
PATTERN = re.compile(
    "|".join(
        re.escape(alias) if not alias.isascii() else rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])"
        for alias in sorted(ALIASES, key=len, reverse=True)
    ),
    re.I,
)
BY_LOWER = {alias.lower(): place for alias, place in ALIASES.items()}


def canonical(place, language):
    if language == "zh":
        return place[1]
    if language == "ms" and place[0] == "Penang":
        return "Pulau Pinang"
    return place[0]


def prepare_terms(text, target):
    return PATTERN.sub(lambda match: canonical(BY_LOWER[match.group().lower()], target), text)


def validate_terms(original, translated, target):
    for match in PATTERN.finditer(original):
        required = canonical(BY_LOWER[match.group().lower()], target)
        pattern = (
            rf"(?<![A-Za-z]){re.escape(required)}(?![A-Za-z])"
            if required.isascii()
            else re.escape(required)
        )
        if not re.search(pattern, translated, re.I):
            raise ValueError("Translation changed a protected place name")
