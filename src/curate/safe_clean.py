"""Local, opt-in payload-preserving TextCleaner (Hinglish Shield local patch).

Credits CuratorKIT issue 42 (https://github.com/Lexsi-Labs/CuratorKIT/issues/42):
default strip_html deletes tag-shaped prompt injections from safety data.

Does NOT disable HTML cleaning globally. Only bare allowlisted formatting tags
are stripped. Attributed tags and their matching closing tags are preserved.
Other configured Unicode/whitespace normalization still applies.
Vendor code is not modified.
"""
from __future__ import annotations

import re

from curatorkit.normalizers.clean import TextCleaner

BENIGN_TAGS = ("b", "i", "u", "em", "strong", "p", "br", "span", "div", "small", "sup", "sub")
_BARE_FORMATTING = re.compile(r"</?(?P<tag>%s)\s*/?>" % "|".join(BENIGN_TAGS), re.IGNORECASE)
_FORMATTING_NAME = re.compile(r"</?(?P<tag>%s)(?=[\s/>])" % "|".join(BENIGN_TAGS), re.IGNORECASE)
_TAG_START = re.compile(r"<\s*/?\s*[|A-Za-z]")


def _tag_spans(text: str):
    """Scan whole tag-shaped spans; quoted literal tags stay inside their owner.

    An unterminated quoted tag conservatively protects the remaining text.
    This scanner preserves prompt payloads; it is not an HTML sanitizer.
    """
    cursor = 0
    while match := _TAG_START.search(text, cursor):
        end = match.end()
        quote = None
        while end < len(text):
            char = text[end]
            end += 1
            if quote is not None:
                if char == quote:
                    quote = None
            elif char in {"'", '"'}:
                quote = char
            elif char == ">":
                break
        yield match.start(), end
        cursor = end


def _protected_names(text: str) -> set[str]:
    """Attributes can carry instructions; preserve both ends of that tag name."""
    protected = set()
    for start, end in _tag_spans(text):
        span = text[start:end]
        name = _FORMATTING_NAME.match(span)
        if name and _BARE_FORMATTING.fullmatch(span) is None:
            protected.add(name.group("tag").lower())
    return protected


def payload_spans(text: str) -> list[str]:
    """Tag-shaped spans that are not benign formatting (potential injection payload)."""
    protected = _protected_names(text or "")
    spans = []
    for start, end in _tag_spans(text or ""):
        span = text[start:end]
        formatting = _BARE_FORMATTING.fullmatch(span)
        if formatting is None or formatting.group("tag").lower() in protected:
            spans.append(span)
    return spans


class PayloadPreservingTextCleaner(TextCleaner):
    def _strip_html(self, text: str) -> str:
        protected = _protected_names(text)
        chunks = []
        cursor = 0
        for start, end in _tag_spans(text):
            chunks.append(text[cursor:start])
            span = text[start:end]
            formatting = _BARE_FORMATTING.fullmatch(span)
            if formatting is None or formatting.group("tag").lower() in protected:
                chunks.append(span)
            cursor = end
        chunks.append(text[cursor:])
        return "".join(chunks)
