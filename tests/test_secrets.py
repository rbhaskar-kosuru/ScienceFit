"""API keys never reach a URL, exception, printed line, log record or tracked file. All keys here are fake."""
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from types import SimpleNamespace

import pytest
import requests
from src import config, curate, llm
from src.redact import KEY_SETTINGS, redact
from tests.conftest import BLANKED
from tests.test_pipeline import FakeResponse, FakeSession, paper

OPENALEX_KEY = "test-openalex-key-123"
LLM_KEY = "test-llm-key-456"
KEYS = (OPENALEX_KEY, LLM_KEY)
LEAKY = (f"HTTPSConnectionPool(host='h'): Max retries exceeded with url: /works?api_key={OPENALEX_KEY} "
         f"headers={{'Authorization': 'Bearer {LLM_KEY}'}}")  # what a raw request error can hold
NETWORK_ERRORS = [requests.ConnectionError(LEAKY), requests.Timeout(LEAKY), requests.exceptions.InvalidHeader(LEAKY),
                  requests.TooManyRedirects(LEAKY), UnicodeEncodeError("latin-1", LEAKY, 0, 1, LEAKY)]
STATUSES = [400, 401, 403, 404, 429, 500, 503]


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch, caplog):
    monkeypatch.setattr(config, "OPENALEX_API_KEY", OPENALEX_KEY)
    monkeypatch.setattr(config, "LLM_API_KEY", LLM_KEY)
    # only curate's clock: patching time.sleep itself would starve Streamlit's threads
    monkeypatch.setattr(curate, "time", SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic))
    caplog.set_level(logging.DEBUG)


def assert_clean(*texts):
    for text in texts:
        assert not any(key in text for key in KEYS), "a key leaked"  # the text itself stays out of the message


def assert_no_leak(error, capsys, caplog):
    out, err = capsys.readouterr()
    shown = "".join(traceback.format_exception(error))
    assert_clean(str(error), repr(error), shown, out, err, caplog.text, *(repr(r.__dict__) for r in caplog.records))


class LeakyResponse(FakeResponse):
    """An error whose body, text and raise_for_status all hold the keys."""
    def __init__(self, status):
        super().__init__({"error": LEAKY}, content=LEAKY.encode(), status=status)
        self.text, self.url = LEAKY, f"https://h/works?api_key={OPENALEX_KEY}"

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Error for url: {self.url}", response=self)


def fail_with(outcome):
    def respond(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return LeakyResponse(outcome)
    return respond


OUTCOMES = pytest.mark.parametrize("outcome", [*NETWORK_ERRORS, *STATUSES], ids=lambda o: getattr(type(o), "__name__", "") + (str(o) if isinstance(o, int) else ""))


# --- Clients ---

@OUTCOMES
def test_openalex_errors(outcome, capsys, caplog):
    client = curate.OpenAlexClient(mailto="", api_key=OPENALEX_KEY, session=FakeSession(fail_with(outcome)))
    with pytest.raises(RuntimeError, match="OpenAlex request failed: ") as e:
        client.search("q")
    assert_no_leak(e.value, capsys, caplog)
    assert_clean(repr(client))
    if isinstance(outcome, int):
        assert f"HTTP {outcome}" in str(e.value)


@OUTCOMES
def test_pubmed_errors(outcome, capsys, caplog):
    session = FakeSession(None)
    session.post = fail_with(outcome)
    with pytest.raises(RuntimeError, match="PubMed") as e:
        curate.PubMedClient(email="", session=session, delay=0).summaries(["1"])
    assert_no_leak(e.value, capsys, caplog)


@OUTCOMES
def test_llm_errors_and_retries(outcome, capsys, caplog, monkeypatch):
    monkeypatch.setattr(llm.requests, "post", fail_with(outcome))
    client = llm.LLMClient(base_url="https://llm.example/v1", model="m", sleep=lambda s: None)
    with pytest.raises(llm.LLMError) as e:
        client.generate("s", "u")
    assert_no_leak(e.value, capsys, caplog)
    assert_clean(repr(client))


@pytest.mark.parametrize("key", ["test-llm-key-456\n", "test-llm-key-456…"])
def test_a_badly_pasted_key_is_not_echoed_by_the_real_http_stack(key, monkeypatch, capsys, caplog):
    monkeypatch.setattr(llm, "LOCAL_HOSTS", set())  # so the key is sent; headers are encoded before connecting
    monkeypatch.setattr(curate, "OPENALEX", "http://127.0.0.1:1/works")
    for call in (lambda: llm.LLMClient(base_url="http://127.0.0.1:1/v1", model="m", api_key=key).generate("s", "u"),
                 lambda: curate.OpenAlexClient(mailto="", api_key=key).search("q")):
        with pytest.raises(RuntimeError, match="InvalidHeader|UnicodeEncodeError") as e:
            call()
        assert_no_leak(e.value, capsys, caplog)


@OUTCOMES
def test_pdf_download_reasons(outcome, tmp_path, capsys, caplog):
    p = curate.Paper(**{**vars(paper("A", "d")), "paper_id": "a_2017", "pdf_urls": [f"https://h/x?api_key={OPENALEX_KEY}"]})
    reason = curate.PDFDownloader(tmp_path, session=FakeSession(fail_with(outcome)), host_delay=0).fetch(p)
    assert reason and not list(tmp_path.iterdir())
    assert_clean(reason, caplog.text, *capsys.readouterr())


def test_non_json_responses_are_safe(capsys, caplog):
    class NotJson(FakeResponse):
        def json(self):
            raise requests.JSONDecodeError(LEAKY, LEAKY, 0)

    session = FakeSession(lambda *a, **k: NotJson())
    session.post = lambda *a, **k: NotJson()
    for call in (lambda: curate.OpenAlexClient(mailto="", api_key=OPENALEX_KEY, session=session).search("q"),
                 lambda: curate.PubMedClient(email="", session=session, delay=0).summaries(["1"])):
        with pytest.raises(RuntimeError, match="not JSON") as e:
            call()
        assert_no_leak(e.value, capsys, caplog)


def test_curate_cli_redacts_any_error(monkeypatch, capsys, caplog):
    def boom(*args, **kwargs):
        raise requests.HTTPError(LEAKY)
    monkeypatch.setattr(curate, "curate", boom)
    monkeypatch.setattr("sys.argv", ["curate"])
    with pytest.raises(SystemExit):
        curate.main()
    out, err = capsys.readouterr()
    assert "Curation failed" in err and "Previous manifest kept" in err
    assert_clean(out, err, caplog.text)


# --- Where the OpenAlex key goes ---

def test_openalex_key_is_a_header_and_never_in_the_url_or_debug_logs(monkeypatch, caplog, capsys):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Length", "15")
            self.end_headers()
            self.wfile.write(b'{"results": []}')

        def log_message(self, *args):
            pass

    with TCPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        monkeypatch.setattr(curate, "OPENALEX", f"http://127.0.0.1:{server.server_address[1]}/works")
        assert curate.OpenAlexClient(mailto="", api_key=OPENALEX_KEY).search("q") == []
        curate.OpenAlexClient(mailto="", api_key="").search("q")
        server.shutdown()
    assert [auth for _, auth in seen] == [f"Bearer {OPENALEX_KEY}", None]
    assert "api_key" not in seen[0][0] and "urllib3" in {r.name.split(".")[0] for r in caplog.records}
    assert_clean(seen[0][0], caplog.text, *capsys.readouterr())


# --- UI and config ---

@pytest.mark.parametrize("outcome, shown", [(requests.ConnectionError(LEAKY), "ConnectionError"), (500, "HTTP 500")])
def test_ui_shows_no_request_details(monkeypatch, outcome, shown):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(requests, "post", fail_with(outcome))
    app = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=20).run()
    app.chat_input[0].set_value("How many sets?").run()
    text = " ".join(m.value for m in app.markdown)
    assert f"API error: {shown}" in text and not app.exception
    assert_clean(text)


def test_bad_extra_params_do_not_echo_their_value():
    run = subprocess.run([sys.executable, "-c", "import src.config"], cwd=config.ROOT, capture_output=True, text=True,
                         env={**os.environ, "LLM_EXTRA_PARAMS": f"{{api_key: {LLM_KEY}"})
    assert run.returncode != 0 and "not valid JSON" in run.stderr
    assert_clean(run.stdout, run.stderr)


def test_tests_never_hold_real_keys_and_every_key_setting_is_known_to_redact():
    out = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print([n for n in dir(c) if n.endswith('_API_KEY')], "
         "any(getattr(c, n) for n in dir(c) if n.endswith('_API_KEY')))"], cwd=config.ROOT, text=True)
    assert out.split("] ")[1].strip() == "False"  # conftest blanked them before .env could load
    assert set(KEY_SETTINGS) <= set(BLANKED)  # whatever .env holds
    assert sorted(KEY_SETTINGS) == sorted(n for n in dir(config) if n.endswith("_API_KEY"))


# --- Redaction ---

@pytest.mark.parametrize("text, clean", [
    (f"key {OPENALEX_KEY} and {LLM_KEY}", "key *** and ***"),
    ("x " + "gsk_" + "a1B2c3D4e5", "x ***"),
    ("x " + "sk-" + "proj-a1B2c3D4e5", "x ***"),
    ("/works?search=q&api_key=abc123&page=2", "/works?search=q&api_key=***&page=2"),
    ('{"api_key": "abc123"}', '{"api_key": "***"}'),
    ("Authorization: Bearer abc.def-123", "Authorization: Bearer ***"),
    ("https://user:pass@host/x", "https://***@host/x"),
    ("risk-adjusted-training-volume and disk-space-checks", "risk-adjusted-training-volume and disk-space-checks"),
    ("OpenAlex request failed: HTTP 429", "OpenAlex request failed: HTTP 429"),
])
def test_redact(text, clean):
    assert redact(text) == clean


def test_redact_handles_url_encoded_and_extra_keys_and_blank_settings(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    assert redact("a%2Fb%2Bc a/b+c", "a/b+c") == "*** ***"
    assert redact("plain text", "") == "plain text"


# --- Repo scan ---

SCAN = [re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9]{20,}"), re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"api_?key[\"']?\s*[=:]\s*[\"']?(?P<value>[A-Za-z0-9_-]{12,})", re.I)]
PLACEHOLDER = re.compile(r"(?:test|fake|your|example)[-_]", re.I)


def key_shaped(line):
    for pattern in SCAN:
        for m in pattern.finditer(line):
            value = m.groupdict().get("value")
            if value is None or (re.search(r"\d", value) and not PLACEHOLDER.match(value)):
                return True
    return False


def test_scan_finds_planted_keys_and_accepts_placeholders():
    assert all(map(key_shaped, ["x=" + "gsk_" + "a" * 24, "sk-" + "b" * 24, "LLM_API_" + "KEY=" + "abc123" * 4,
                                '"api_' + 'key": "' + "Zx9" * 6 + '"']))
    assert not any(map(key_shaped, ["LLM_API_KEY=", "LLM_API_KEY=your-key", "api_key=test-openalex-key-123",
                                    "api_key=config.OPENALEX_API_KEY", "risk-adjusted-training-volume-load"]))


def test_no_tracked_file_contains_a_key():
    """Also covers new files that aren't ignored, so a leak fails before it's committed. Reports places, never values."""
    try:
        listed = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=config.ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    files = [f for f in listed.split("\0") if f and (config.ROOT / f).is_file()]
    assert "src/config.py" in files and ".env" not in files
    found = [f"{f}:{n}" for f in files
             for n, line in enumerate((config.ROOT / f).read_text(errors="replace").splitlines(), 1) if key_shaped(line)]
    assert found == []


def test_env_example_has_no_key_values():
    lines = (config.ROOT / ".env.example").read_text().splitlines()
    values = [line.split("=", 1)[1].strip() for line in lines if re.match(r"#?\s*\w*API_KEY=", line)]
    assert len(values) == len(KEY_SETTINGS) and all(v == "" or PLACEHOLDER.match(v) for v in values)
