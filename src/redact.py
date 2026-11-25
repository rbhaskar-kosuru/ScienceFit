"""Safety net: removes API keys from text before it's printed, logged or put in an exception. Clients should not put
keys in such text in the first place."""
import re
from urllib.parse import quote

from . import config

MASK = "***"
KEY_SETTINGS = ("LLM_API_KEY", "OPENALEX_API_KEY")  # config names that hold a key; add new ones here
KEY_SHAPED = [
    (re.compile(r"(?<![A-Za-z0-9])(?:gsk_|sk-)[A-Za-z0-9_-]{8,}"), MASK),
    (re.compile(r"(\bBearer\s+)[^\s\"',}]+", re.I), rf"\1{MASK}"),
    (re.compile(r"((?:api[_-]?key|access[_-]?token)[\"']?\s*[=:]\s*[\"']?)[^\s&\"',}#]+", re.I), rf"\1{MASK}"),
    (re.compile(r"(://)[^/\s@]+@"), rf"\1{MASK}@"),  # user:password in a URL
]


def redact(text: object, *keys: str) -> str:
    """`text` without the configured keys, any extra `keys` (also URL-encoded) and anything key-shaped."""
    text = str(text)
    known = {k for k in (*keys, *(getattr(config, name, "") for name in KEY_SETTINGS)) if k}
    for key in sorted(known | {quote(k, safe="") for k in known}, key=len, reverse=True):
        text = text.replace(key, MASK)
    for pattern, replacement in KEY_SHAPED:
        text = pattern.sub(replacement, text)
    return text
