"""Local, opt-in payload-preserving TextCleaner (Hinglish Shield local patch).

Credits CuratorKIT issue 42 (https://github.com/Lexsi-Labs/CuratorKIT/issues/42):
default strip_html deletes tag-shaped prompt injections from safety data.

Does NOT disable HTML cleaning globally. Only a small allowlist of benign
formatting tags is stripped; every other tag-shaped span (<system>, <admin>,
<script>, <|im_start|>, unknown tags) is preserved verbatim.
Vendor code is not modified.
"""
from __future__ import annotations

import re

from curatorkit.normalizers.clean import TextCleaner

BENIGN_TAGS = ("b", "i", "u", "em", "strong", "p", "br", "span", "div", "small", "sup", "sub")
_BENIGN_RE = re.compile(r"</?(?:%s)(?:\s[^<>]*)?/?>" % "|".join(BENIGN_TAGS), re.IGNORECASE)
PAYLOAD_RE = re.compile(r"<\s*/?\s*[|A-Za-z][^<>]{0,200}>")


def payload_spans(text: str) -> list[str]:
    """Tag-shaped spans that are not benign formatting (potential injection payload)."""
    return [m.group(0) for m in PAYLOAD_RE.finditer(text or "") if not _BENIGN_RE.fullmatch(m.group(0))]


class PayloadPreservingTextCleaner(TextCleaner):
    def _strip_html(self, text: str) -> str:
        return _BENIGN_RE.sub("", text)
