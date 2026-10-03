import re
import unicodedata
from html import unescape

from bs4 import BeautifulSoup
from ftfy import fix_text

BLOCK_TAGS = {"p", "div", "section", "article", "li", "blockquote", "h1", "h2", "h3", "tr"}


def clean_text(value: str | None, *, single_line: bool = False) -> str | None:
    """Repair encoding, extract visible text, and remove adjacent repeated paragraphs."""
    if value is None:
        return None
    value = fix_text(
        value,
        unescape_html=False,
        uncurl_quotes=False,
        fix_latin_ligatures=False,
        fix_character_width=False,
        normalization="NFC",
    )
    # Repeatedly escaped feed HTML is common. Stop once decoding stabilizes.
    for _ in range(5):
        decoded = unescape(value)
        if decoded == value:
            break
        value = decoded
    if re.search(r"</?[a-zA-Z][^>]*>|<!--", value):
        soup = BeautifulSoup(value, "html.parser")
        for tag in soup(["script", "style", "template", "noscript", "iframe", "svg"]):
            tag.decompose()
        for tag in soup.find_all("br"):
            tag.replace_with("\n")
        for tag in soup.find_all(BLOCK_TAGS):
            tag.insert_before("\n")
            tag.insert_after("\n")
        value = soup.get_text()
    value = "".join(
        "\ufffd" if unicodedata.category(char) == "Cs" else char
        for char in value
        if (unicodedata.category(char) != "Cc" or char in "\n\t")
        and char not in "\u200b\ufeff\u2060\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    )
    paragraphs = []
    for line in value.splitlines():
        paragraph = re.sub(r"\s+", " ", line).strip()
        if not paragraph:
            continue
        if paragraphs and len(paragraph) >= 20 and paragraph == paragraphs[-1]:
            continue
        paragraphs.append(paragraph)
    result = (" " if single_line else "\n\n").join(paragraphs)
    return result or None


def meaningful(value: str | None) -> bool:
    """Keep short titles and non-Latin scripts; reject markup/punctuation-only text."""
    return value is not None and any(char.isalnum() for char in value)
