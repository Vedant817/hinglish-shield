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
_BENIGN_RE = re.compile(r"</?(?P<tag>%s)(?:\s[^<>]*)?/?>" % "|".join(BENIGN_TAGS), re.IGNORECASE)
_FORMATTING_OPEN = re.compile(r"<(?P<tag>%s)(?=[\s/>])(?P<suffix>[^>]*)>" % "|".join(BENIGN_TAGS), re.IGNORECASE)
PAYLOAD_RE = re.compile(r"<\s*/?\s*[|A-Za-z][^<>]{0,200}>")


def _protected_names(text: str) -> set[str]:
    """Attributes can carry instructions; preserve both ends of that tag name."""
    return {m.group("tag").lower() for m in _FORMATTING_OPEN.finditer(text)
            if m.group("suffix").strip() not in {"", "/"}}


def payload_spans(text: str) -> list[str]:
    """Tag-shaped spans that are not benign formatting (potential injection payload)."""
    protected = _protected_names(text or "")
    spans = []
    for match in PAYLOAD_RE.finditer(text or ""):
        formatting = _BENIGN_RE.fullmatch(match.group(0))
        if formatting is None or formatting.group("tag").lower() in protected:
            spans.append(match.group(0))
    return spans


class PayloadPreservingTextCleaner(TextCleaner):
    def _strip_html(self, text: str) -> str:
        protected = _protected_names(text)
        return _BENIGN_RE.sub(
            lambda match: match.group(0) if match.group("tag").lower() in protected else "", text,
        )
