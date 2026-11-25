"""Generic LLM client: any OpenAI-compatible chat-completions API. Switching provider is a .env change only."""
import re
import sys
import time
from urllib.parse import urlsplit

import requests

from . import config
from .redact import redact

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}  # never sent the API key
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
THINK_END = re.compile(r"^.*</think>", re.S | re.I)   # opening tag missing: some chat templates swallow it
THINK_START = re.compile(r"<think>.*$", re.S | re.I)  # never closed: the output was cut off mid-reasoning


class LLMError(RuntimeError):
    """Never carries request headers or the API key."""


class LLMClient:
    """Unset arguments come from config (i.e. .env)."""
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 extra_params: dict | None = None, sleep=time.sleep):
        self.url = (base_url or config.LLM_BASE_URL).rstrip("/") + "/chat/completions"
        self.model = model or config.LLM_MODEL
        self.extra_params = config.LLM_EXTRA_PARAMS if extra_params is None else extra_params
        self._api_key = config.LLM_API_KEY if api_key is None else api_key
        self._sleep = sleep

    def generate(self, system: str, prompt: str) -> str:
        data = self._post({
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
            **self.extra_params,
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
        })
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""  # any separate reasoning field is ignored
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError(f"LLM {self.model}: unexpected response shape") from None
        answer = strip_reasoning(content)
        if not answer:
            raise LLMError(f"LLM {self.model}: empty answer (finish_reason={choice.get('finish_reason')})")
        return answer

    def _post(self, payload: dict) -> dict:
        send_key = self._api_key and urlsplit(self.url).hostname not in LOCAL_HOSTS
        headers = {"Authorization": f"Bearer {self._api_key}"} if send_key else {}
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                r = requests.post(self.url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT,
                                  allow_redirects=False)
            except (requests.RequestException, ValueError) as e:  # only the type: the exception holds the request
                raise LLMError(f"LLM {self.model}: request failed ({type(e).__name__})") from None
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    raise LLMError(f"LLM {self.model}: response is not JSON") from None
            delay = retry_delay(r, attempt)
            retryable = r.status_code == 429 or r.status_code >= 500
            if not retryable or attempt == config.LLM_MAX_RETRIES \
                    or delay > config.LLM_MAX_RETRY_WAIT:
                raise LLMError(f"LLM {self.model}: HTTP {r.status_code}{self._detail(r)}")
            print(f"LLM HTTP {r.status_code}, retry {attempt + 1}/{config.LLM_MAX_RETRIES} in {delay:.1f}s",
                  file=sys.stderr)
            self._sleep(delay)

    def _detail(self, r: requests.Response) -> str:
        """Provider's error text, key redacted. Dropped for auth errors, where some providers echo part of the key."""
        if r.status_code in (401, 403):
            return " (check LLM_API_KEY)"
        text = redact(r.text, self._api_key)
        return f": {text[:300]}" if text else ""


def retry_delay(r: requests.Response, attempt: int) -> float:
    """Retry-After in seconds if the provider sent one, else 1, 2, 4, 8… seconds."""
    try:
        return max(0.0, float(r.headers["Retry-After"]))
    except (KeyError, ValueError):  # missing, or the HTTP-date form
        return float(2 ** attempt)


def strip_reasoning(text: str) -> str:
    for pattern in (THINK_BLOCK, THINK_END, THINK_START):
        text = pattern.sub("", text)
    return text.strip()


def generate(system: str, prompt: str) -> str:
    return LLMClient().generate(system, prompt)
