"""E-mail cleanup (FR-16): keep the new text, drop quoted history, signature and footers.

quotequail handles `>`-quoting and English reply headers. Russian clients write reply
headers it does not know («пн, 14 сент. 2026 г. в 10:03, Имя <адрес>:», «-----Исходное
сообщение-----», Outlook's «От: … Отправлено: …»), so those are matched here first.
"""

from __future__ import annotations

import email
import re
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime

import quotequail
from pydantic import BaseModel

MONTHS = r"(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\.?"
REPLY_HEADERS = [
    # Gmail / Mail.ru / Yandex: "пн, 14 сент. 2026 г. в 10:03, Имя <addr>:" (may wrap onto two lines)
    re.compile(
        rf"^\s*(?:[а-я]{{2}},\s*)?\d{{1,2}}\s+{MONTHS}\s+\d{{4}}\s*г?\.?,?\s*(?:в\s*)?\d{{1,2}}:\d{{2}}.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^\s*-{2,}\s*(?:Исходное|Пересылаемое|Original)\s+(?:сообщение|message)\s*-{2,}\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Outlook block: "От: ..." followed within a few lines by "Отправлено:" or "Дата:"
    re.compile(
        r"^\s*От(?:правитель)?:\s.+\n(?:.*\n){0,2}?\s*(?:Отправлено|Дата|Sent):", re.IGNORECASE | re.MULTILINE
    ),
]
SIGNATURE = [
    re.compile(r"^\s*--\s*$", re.MULTILINE),
    re.compile(
        r"^\s*(?:С\s+уважением|С\s+наилучшими\s+пожеланиями|Всего\s+доброго|Best\s+regards)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*(?:Отправлено\s+(?:с|из)\s+|Sent\s+from\s+my\s+).*$", re.IGNORECASE | re.MULTILINE),
]


class CleanEmail(BaseModel):
    sender: str
    sender_name: str
    to: str
    subject: str
    date: str | None  # ISO 8601
    body: str
    removed: dict[str, str]  # part -> removed text, kept for audit


def clean_email(raw: bytes) -> CleanEmail:
    msg = email.message_from_bytes(raw)
    text = _plain_text(msg)
    removed: dict[str, str] = {}

    body, quoted = _cut_first(text, REPLY_HEADERS)
    if quoted:
        removed["quoted"] = quoted
    # Remaining '>' quoting and English reply headers.
    parts = quotequail.quote(body)
    body = "".join(t for keep, t in parts if keep)
    rest = "".join(t for keep, t in parts if not keep)
    if rest.strip():
        removed["quoted"] = (rest + removed.get("quoted", "")).strip()

    body, signature = _cut_first(body, SIGNATURE)
    if signature:
        removed["signature"] = signature

    name, addr = parseaddr(_header(msg, "From"))
    date = msg.get("Date")
    return CleanEmail(
        sender=addr,
        sender_name=name,
        to=parseaddr(_header(msg, "To"))[1],
        subject=_header(msg, "Subject"),
        date=parsedate_to_datetime(date).isoformat() if date else None,
        body=_tidy(body),
        removed={k: v.strip() for k, v in removed.items()},
    )


def _cut_first(text: str, patterns: list[re.Pattern[str]]) -> tuple[str, str]:
    """Split at the earliest match of any pattern: (before, from-match-on)."""
    starts = [m.start() for p in patterns if (m := p.search(text))]
    if not starts:
        return text, ""
    cut = min(starts)
    return text[:cut], text[cut:]


def _plain_text(msg: Message) -> str:
    part = next((p for p in msg.walk() if p.get_content_type() == "text/plain"), None)
    if part is None:
        raise ValueError("no text/plain part")
    payload = part.get_payload(decode=True)
    return payload.decode(part.get_content_charset() or "utf-8", errors="replace").replace("\r\n", "\n")


def _header(msg: Message, name: str) -> str:
    value = msg.get(name)
    return str(make_header(decode_header(value))) if value else ""


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
