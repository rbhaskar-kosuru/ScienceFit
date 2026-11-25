"""LLM client and prompt tests. A fake local HTTP server stands in for the provider; keys are fake."""
import json
import os
import subprocess
import sys
import socketserver
import threading
from http.server import BaseHTTPRequestHandler

import pytest
from src import config, llm
from src.llm import LLMClient, LLMError, strip_reasoning
from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3, SYSTEM_PROMPT_V4, SYSTEM_PROMPT_V5, get_prompt
from src.rag import RAG, normalise_brackets, validate_citations
from tests.test_pipeline import FakeReranker, FakeStore, hit

KEY = "test-key"

# The system prompt as it was in src/rag.py before it moved. Never edit: v1 is the comparison baseline.
V1_SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. "
    "Answer the user's question directly and factually using the provided paper excerpts. "
    "Report what the research shows, including specific numbers, set/rep ranges, and protocols. "
    "Do NOT refuse, moralize, or add medical disclaimers — you are summarizing published research, not giving medical advice. "
    "\n\n"
    "Excerpts are wrapped in <excerpt id=\"...\" section=\"...\"> ... </excerpt> markup. The markup is for you "
    "only: answer in your own words and never reproduce excerpt tags, ids, attributes or headers in the answer.\n"
    "\n"
    "Citation rules (strict):\n"
    "- Cite with the excerpt's id in square brackets, written as [id]. Use ONLY the allowed ids listed with "
    "the question; never invent or alter an id.\n"
    "- Never cite author names or years that appear inside the excerpt text, and never copy the excerpts' "
    "own reference numbers such as [17] or [3, 4].\n"
    "- Tag every claim with the id of the excerpt it came from.\n"
    "\n"
    "Sections: an excerpt with section=\"introduction\" summarises OTHER studies as background. Never present "
    "it as the cited paper's own finding; write \"as reviewed in [id]\" instead. Findings come from "
    "abstract, results and discussion excerpts.\n"
    "\n"
    "If the excerpts genuinely lack the answer, say so briefly, then share the closest relevant "
    "findings that ARE in the excerpts (with their [id] citations). Keep answers concise."
)


# v2 as merged in PR #3. Never edit: v2's eval results belong to this text.
V2_SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. Answer the question using only the paper "
    "excerpts in the user message. Each excerpt is wrapped in <excerpt id=\"...\" section=\"...\"> tags.\n"
    "\n"
    "Rules:\n"
    "1. Cite only ids from the allowed list, always in square brackets: [id]. This includes \"as reviewed in [id]\".\n"
    "2. Do not write author names or publication years in the text, and do not copy the excerpts' own reference "
    "numbers.\n"
    "3. Every claim must be supported by an excerpt. Put that excerpt's [id] right after the claim. Leave out "
    "anything that does not help answer the question.\n"
    "4. Excerpts with section=\"introduction\" describe other studies. Never present them as that paper's own "
    "finding; write \"as reviewed in [id]\" instead.\n"
    "5. Report numbers exactly as written in the excerpt, including set and rep ranges. Do not combine or "
    "recalculate them.\n"
    "6. If the excerpts do not answer the question, say so briefly, then give the closest relevant findings that "
    "are in the excerpts.\n"
    "\n"
    "Output only the answer, concisely and in your own words. No preamble, no reasoning, no excerpt tags. "
    "Do not moralize or add medical disclaimers: you are summarizing published research."
)


# v3 as evaluated. Never edit.
V3_SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. Answer the question using only the paper "
    "excerpts in the user message. Each excerpt is wrapped in <excerpt id=\"...\" section=\"...\"> tags.\n"
    "\n"
    "Rules:\n"
    "1. Cite only ids from the allowed list, always in square brackets: [id]. This includes \"as reviewed in [id]\".\n"
    "2. Do not write author names or publication years in the text, and do not copy the excerpts' own reference "
    "numbers.\n"
    "3. Every claim must be supported by an excerpt. Put that excerpt's [id] right after the claim. Leave out "
    "anything that does not help answer the question.\n"
    "4. Excerpts with section=\"introduction\" describe other studies. Never present them as that paper's own "
    "finding; write \"as reviewed in [id]\" instead.\n"
    "5. Report numbers exactly as written in the excerpt, including set and rep ranges. Do not combine or "
    "recalculate them.\n"
    "6. If the excerpts do not answer the question, say so briefly, then give the closest relevant findings that "
    "are in the excerpts.\n"
    "7. Start with a one-sentence direct answer to the question: yes, no, it depends, or the excerpts don't say.\n"
    "8. Then give the key evidence in plain language, for a reader who is not a scientist.\n"
    "9. Include a number only if you say what it measures in everyday words. Skip technical units and statistics "
    "(µm², CI, p-values, effect sizes) unless the question asks for them.\n"
    "10. Keep it short: 3 to 5 sentences unless the question needs more.\n"
    "\n"
    "Output only the answer, concisely and in your own words. No preamble, no reasoning, no excerpt tags. "
    "Do not moralize or add medical disclaimers: you are summarizing published research."
)


# v4 as evaluated. Never edit.
V4_SYSTEM_PROMPT = (
    "You are ScienceFit, a research assistant for resistance training. Answer the question using only the paper "
    "excerpts in the user message. Each excerpt is wrapped in <excerpt id=\"...\" section=\"...\"> tags.\n"
    "\n"
    "Rules:\n"
    "1. Cite only ids from the allowed list, always in square brackets: [id]. This includes \"as reviewed in [id]\".\n"
    "2. Do not write author names or publication years in the text, and do not copy the excerpts' own reference "
    "numbers.\n"
    "3. Every claim must be supported by an excerpt. Put that excerpt's [id] right after the claim. Leave out "
    "anything that does not help answer the question.\n"
    "4. Excerpts with section=\"introduction\" describe other studies. So does any excerpt, whatever its section, "
    "where it reports what another study found. Never present these as that paper's own finding; write \"as reviewed "
    "in [id]\" instead.\n"
    "5. Use only numbers that appear in an excerpt. Rounding is fine. Never multiply, add or combine numbers (for "
    "example sets times sessions), and never merge separate ranges.\n"
    "6. If the excerpts do not answer the question, say so briefly, then give the closest relevant findings that "
    "are in the excerpts.\n"
    "7. Start with a one-sentence direct answer to the question: yes, no, it depends, or the excerpts don't say. "
    "It must be fully supported by the evidence that follows and end with a citation. If the excerpts disagree or "
    "are mixed, open with \"It depends\" or \"The evidence is mixed\", not \"Yes\" or \"No\".\n"
    "8. If the question asks whether X increases or improves Y, answer whether X beats the usual alternative (such "
    "as not doing X), not whether X works at all.\n"
    "9. Then give the key evidence in plain language, for a reader who is not a scientist. Include a number only if "
    "you say what it measures in everyday words.\n"
    "10. Do not use microscopic or lab units or statistics (µm², mm², mmol, CI, p-values, effect sizes). Say the "
    "result in words or leave it out.\n"
    "11. If an excerpt gives a practical limit or threshold, such as an amount beyond which there is no further "
    "gain, include it when it is relevant to the question.\n"
    "12. Keep it short: 3 to 5 sentences unless the question needs more.\n"
    "\n"
    "Output only the answer, concisely and in your own words. No preamble, no reasoning, no excerpt tags. "
    "Do not moralize or add medical disclaimers: you are summarizing published research."
)


def chat(content, **message):
    return 200, {"choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": "stop"}]}


@pytest.fixture
def server():
    """Replies with server.replies in order: (status, body[, headers]). Records requests in server.seen."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.seen.append((self.path, self.headers.get("Authorization"), body))
            status, reply, *headers = self.server.replies.pop(0)
            data = reply.encode() if isinstance(reply, str) else json.dumps(reply).encode()
            self.send_response(status)
            for name, value in (headers[0] if headers else {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)  # HTTPServer does a slow reverse-DNS lookup on macOS
    srv.seen, srv.replies = [], []
    threading.Thread(target=srv.serve_forever, args=(0.01,), daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def client(server, **kw):
    sleeps = []
    c = LLMClient(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", model="m", sleep=sleeps.append,
                  **{"api_key": KEY, "extra_params": {}, **kw})
    c.sleeps = sleeps
    return c


@pytest.fixture
def remote(monkeypatch):
    """Treat the fake server on 127.0.0.1 as a hosted provider."""
    monkeypatch.setattr(llm, "LOCAL_HOSTS", set())


def test_client_sends_a_standard_chat_completion(server, remote):
    server.replies = [chat("Yes [a_2020].")]
    assert client(server, extra_params={"reasoning_effort": "low"}).generate("sys", "user") == "Yes [a_2020]."
    path, auth, body = server.seen[0]
    assert path == "/v1/chat/completions" and auth == f"Bearer {KEY}"
    assert body == {"model": "m", "temperature": 0, "seed": 42, "stream": False, "reasoning_effort": "low",
                    "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]}


def test_client_sends_no_auth_header_without_a_key(server, remote):
    server.replies = [chat("ok")]
    client(server, api_key="").generate("s", "u")
    assert server.seen[0][1] is None


def test_key_is_not_sent_to_a_local_server(server):
    server.replies = [chat("ok")]
    client(server).generate("s", "u")  # 127.0.0.1, key set
    assert server.seen[0][1] is None


@pytest.mark.parametrize("base_url, sent", [
    ("http://localhost:11434/v1", False), ("http://LOCALHOST/v1", False), ("http://127.0.0.1:11434/v1", False),
    ("http://[::1]:11434/v1", False), ("https://api.example.com/v1", True), ("https://localhost.example.com/v1", True),
])
def test_key_is_only_sent_to_non_local_hosts(monkeypatch, base_url, sent):
    calls = []

    class Reply:
        status_code = 200
        def json(self):
            return chat("ok")[1]

    monkeypatch.setattr(llm.requests, "post", lambda url, headers, **kw: calls.append(headers) or Reply())
    LLMClient(base_url=base_url, model="m", api_key=KEY, extra_params={}).generate("s", "u")
    assert calls == [{"Authorization": f"Bearer {KEY}"} if sent else {}]


def test_client_retries_429_and_5xx_with_backoff_and_retry_after(server, remote, capsys):
    server.replies = [(429, "slow down", {"Retry-After": "7"}), (429, "slow down"), (503, "busy"), chat("ok")]
    c = client(server)
    assert c.generate("s", "u") == "ok"
    assert c.sleeps == [7.0, 2.0, 4.0] and len(server.seen) == 4
    err = capsys.readouterr().err
    assert "HTTP 429" in err and KEY not in err and "Authorization" not in err


def test_client_gives_up_after_max_retries_and_on_long_waits(server):
    server.replies = [(429, "no")] * (config.LLM_MAX_RETRIES + 1)
    with pytest.raises(LLMError, match="HTTP 429"):
        client(server).generate("s", "u")
    assert len(server.seen) == config.LLM_MAX_RETRIES + 1
    server.seen.clear()
    server.replies = [(429, "daily quota", {"Retry-After": "3600"})]
    with pytest.raises(LLMError, match="daily quota"):
        client(server).generate("s", "u")
    assert len(server.seen) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried_and_never_contain_the_key(server, remote, status):
    server.replies = [(status, f"bad request with {KEY}")]
    c = client(server)
    with pytest.raises(LLMError) as e:
        c.generate("s", "u")
    assert len(server.seen) == 1 and c.sleeps == []
    assert str(status) in str(e.value) and KEY not in str(e.value) + repr(e.value) and "Authorization" not in str(e.value)


def test_connection_errors_do_not_expose_the_request():
    with pytest.raises(LLMError) as e:
        LLMClient(base_url="http://127.0.0.1:1/v1", model="m", api_key=KEY, extra_params={}).generate("s", "u")
    assert KEY not in str(e.value) and e.value.__suppress_context__


def test_empty_answer_is_an_error(server):
    server.replies = [chat(None, reasoning="thinking [a_2020]"), chat("<think>only thoughts</think>")]
    for _ in range(2):
        with pytest.raises(LLMError, match="empty answer"):
            client(server).generate("s", "u")


@pytest.mark.parametrize("raw, clean", [
    ("<think>hmm [x_2020]\n</think>\nAnswer [a_2020].", "Answer [a_2020]."),
    ("<THINK>a</THINK>One <think>b</think>two", "One two"),
    ("hmm [x_2020]</think>Answer.", "Answer."),
    ("Answer.<think>cut off [x_2020]", "Answer."),
    ("No reasoning [a_2020].", "No reasoning [a_2020]."),
])
def test_strip_reasoning(raw, clean):
    assert strip_reasoning(raw) == clean


def test_reasoning_never_reaches_the_citation_validator(server):
    server.replies = [chat("<think>maybe [b_2021] or [fake_1999]</think>Yes [a_2020].", reasoning="see [b_2021]")]
    rag = RAG(FakeStore([hit("a_2020"), hit("b_2021")]), FakeReranker([3.0, 2.0]), client(server).generate)
    res = rag.ask("How many sets?")
    assert res["answer"] == "Yes [a_2020]." and res["unverified_citations"] == []
    assert {s["paper_id"]: s["cited"] for s in res["sources"]} == {"a_2020": True, "b_2021": False}


def test_normalise_brackets():
    text = "One【a_2020】, two 【a_2020; b_2021】, plain [a_2020], (95% CI)."
    assert normalise_brackets(text) == "One[a_2020], two [a_2020; b_2021], plain [a_2020], (95% CI)."
    assert validate_citations(normalise_brackets("【a_2020】 【fake_1999】"), {"a_2020"}) == ({"a_2020"}, ["fake_1999"])


def test_ask_validates_and_returns_normalised_brackets():
    res = RAG(FakeStore([hit("a_2020")]), FakeReranker([3.0]), lambda s, p: "Yes【a_2020】.").ask("How many sets?")
    assert res["answer"] == "Yes[a_2020]." and res["sources"][0]["cited"] and not res["no_citations"]


def test_prompt_v2_is_selectable_and_has_no_example_ids():
    import re
    sent = []
    hits = [hit("b_2021", 0, section="introduction"), hit("a_2020"), hit("b_2021", 1)]
    RAG(FakeStore(hits), FakeReranker([9.0, 5.0, 4.0]), lambda s, p: sent.append((s, p)) or "x [a_2020]",
        prompt_version="v2").ask("How many sets?")
    system, user = sent[0]
    assert system == SYSTEM_PROMPT_V2 != SYSTEM_PROMPT
    assert not re.search(r"\b[a-z][a-z0-9]*_\d{4}", system, re.I)
    assert "Allowed ids: b_2021, a_2020\n" in user
    assert '<excerpt id="b_2021" section="introduction">\nexcerpt from b_2021\n</excerpt>' in user


def test_prompt_v2_is_identical_to_the_merged_prompt():
    assert SYSTEM_PROMPT_V2 == V2_SYSTEM_PROMPT == get_prompt("v2").system


def test_prompt_v3_keeps_v2_rules_and_has_no_example_ids():
    import re
    v3 = get_prompt("v3")
    assert v3.system == SYSTEM_PROMPT_V3 and v3.build is get_prompt("v2").build
    assert all(line in SYSTEM_PROMPT_V3.split("\n") for line in SYSTEM_PROMPT_V2.split("\n"))
    assert "one-sentence direct answer" in SYSTEM_PROMPT_V3 and "plain language" in SYSTEM_PROMPT_V3
    assert not re.search(r"\b[a-z][a-z0-9]*_\d{4}", SYSTEM_PROMPT_V3, re.I)


def test_prompt_v3_is_identical_to_the_evaluated_prompt():
    assert SYSTEM_PROMPT_V3 == V3_SYSTEM_PROMPT == get_prompt("v3").system


def test_prompt_v4_exists_and_has_no_example_ids():
    import re
    v4 = get_prompt("v4")
    assert v4.system == SYSTEM_PROMPT_V4 and v4.build is get_prompt("v2").build
    assert "end with a citation" in SYSTEM_PROMPT_V4 and "Never multiply, add or combine" in SYSTEM_PROMPT_V4
    assert not re.search(r"\b[a-z][a-z0-9]*_\d{4}", SYSTEM_PROMPT_V4, re.I)


def test_prompt_v4_is_identical_to_the_evaluated_prompt():
    assert SYSTEM_PROMPT_V4 == V4_SYSTEM_PROMPT == get_prompt("v4").system


def test_prompt_v5_is_v2_plus_one_rule():
    import re
    added = [line for line in SYSTEM_PROMPT_V5.split("\n") if line not in SYSTEM_PROMPT_V2.split("\n")]
    assert [line for line in SYSTEM_PROMPT_V5.split("\n") if line not in added] == SYSTEM_PROMPT_V2.split("\n")
    assert len(added) == 1 and added[0].startswith('7. Excerpts marked secondhand="true"')
    assert "as reviewed in [id]" in added[0] and get_prompt("v5").system == SYSTEM_PROMPT_V5
    assert not re.search(r"\b[a-z][a-z0-9]*_\d{4}", SYSTEM_PROMPT_V5, re.I)


FLAGGED = [hit("a_2020", section="discussion", secondhand=True), hit("b_2021", section="results", secondhand=False),
           hit("c_2022", secondhand=True), hit("d_2023", section="methods")]


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_older_prompts_render_excerpts_exactly_as_before(version):
    user = get_prompt(version).build("q?", FLAGGED)
    assert user.startswith('<excerpt id="a_2020" section="discussion">\nexcerpt from a_2020\n</excerpt>\n\n'
                           '<excerpt id="b_2021" section="results">\nexcerpt from b_2021\n</excerpt>\n\n'
                           '<excerpt id="c_2022">\nexcerpt from c_2022\n</excerpt>\n\n'
                           '<excerpt id="d_2023" section="methods">\nexcerpt from d_2023\n</excerpt>\n\nQuestion: q?\n\n')
    assert "secondhand" not in user + get_prompt(version).system
    assert user == get_prompt(version).build("q?", [{k: v for k, v in h.items() if k != "secondhand"} for h in FLAGGED])


def test_v5_tags_only_flagged_chunks_and_changes_nothing_else():
    user = get_prompt("v5").build("q?", FLAGGED)
    assert re_tags(user) == ['<excerpt id="a_2020" section="discussion" secondhand="true">', '<excerpt id="b_2021" section="results">',
                             '<excerpt id="c_2022" secondhand="true">', '<excerpt id="d_2023" section="methods">']
    assert user.replace(' secondhand="true"', "") == get_prompt("v2").build("q?", FLAGGED)


def re_tags(text):
    import re
    return re.findall(r"<excerpt [^>]*>", text)


def test_smoke_report_marks_secondhand_excerpts_for_v5():
    from evals.run_smoke import check, excerpts, report
    shown = excerpts(get_prompt("v5").build("q?", FLAGGED))
    assert [(e["id"], e["section"], e["secondhand"]) for e in shown] == [
        ("a_2020", "discussion", True), ("b_2021", "results", False), ("c_2022", "unknown", True), ("d_2023", "methods", False)]
    res = {"answer": "x", "refused": False, "sources": [], "unverified_citations": []}
    out = report({"id": "Q1", "question": "q", "expect_any": ["a_2020"]}, res, check({"expect_any": ["a_2020"]}, res), shown)
    assert "[a_2020] section: discussion secondhand\n" in out and "[b_2021] section: results\n" in out


def test_smoke_eval_counts_answers_and_flags_errors_and_empty_ones(monkeypatch, capsys):
    from evals import run_smoke
    cases = len(json.loads(run_smoke.CASES.read_text()))
    broken = {}

    class FakeRAG:
        def __init__(self, generate, prompt_version):
            self.n = 0

        def ask(self, question):
            self.n += 1
            if broken.get(self.n) == "error":
                raise LLMError("HTTP 500 api_key=abc123 Bearer abc123")
            return {"answer": "  " if broken.get(self.n) == "empty" else "x", "refused": True, "sources": [], "unverified_citations": []}

    monkeypatch.setattr("src.rag.RAG", FakeRAG)
    run_smoke.main([])
    assert f"Answered {cases} of {cases}\n" in capsys.readouterr().out
    broken.update({2: "error", 4: "empty"})
    with pytest.raises(SystemExit) as e:
        run_smoke.main([])
    out = capsys.readouterr().out
    assert e.value.code == 1 and f"Answered {cases - 2} of {cases}  NOT ANSWERED: Q2, Q4\n" in out
    assert "Q2  ERROR" in out and "empty answer" in out and "abc123" not in out and "Q9  " in out


def test_smoke_report_shows_the_excerpts_the_model_saw():
    from evals.run_smoke import Recorder, check, excerpts, report

    class Client:
        def generate(self, system, prompt):
            return "Yes [a_2020]."

    for version in ("v1", "v2"):
        rec = Recorder(Client())
        res = RAG(FakeStore([hit("a_2020", section="results"), hit("b_2021")]), FakeReranker([3.0, 2.0]), rec.generate,
                  prompt_version=version).ask("How many sets?")
        shown = excerpts(rec.prompt)
        assert shown == [{"id": "a_2020", "section": "results", "secondhand": False, "text": "excerpt from a_2020"},
                         {"id": "b_2021", "section": "unknown", "secondhand": False, "text": "excerpt from b_2021"}]
        case = {"id": "Q1", "question": "q", "expect_any": ["a_2020"]}
        out = report(case, res, check(case, res), shown)
        assert "    excerpts:   2\n      [a_2020] section: results\n        excerpt from a_2020" in out


def test_prompt_v1_is_identical_to_the_old_prompt():
    assert SYSTEM_PROMPT == V1_SYSTEM_PROMPT == get_prompt("v1").system
    assert get_prompt("v1").build("q?", [hit("a_2020", section="results"), hit("b_2021")]) == (
        '<excerpt id="a_2020" section="results">\nexcerpt from a_2020\n</excerpt>\n\n'
        '<excerpt id="b_2021">\nexcerpt from b_2021\n</excerpt>\n\nQuestion: q?\n\n'
        "Allowed citation ids: a_2020, b_2021\n"
        "Answer in your own words. Cite only with an allowed id in the format [id]; don't reproduce the excerpt markup.")


def test_rag_sends_the_selected_prompt_and_rejects_unknown_versions():
    sent = []
    rag = RAG(FakeStore([hit("a_2020")]), FakeReranker([3.0]), lambda s, p: sent.append((s, p)) or "Yes [a_2020].")
    rag.ask("How many sets?")
    assert sent == [(V1_SYSTEM_PROMPT, get_prompt("v1").build("How many sets?", [hit("a_2020")]))]
    with pytest.raises(ValueError, match="Unknown prompt version"):
        RAG(FakeStore([]), FakeReranker([]), None, prompt_version="v99")


def test_blank_env_gives_the_local_ollama_defaults():
    env = {**os.environ, "LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_API_KEY": "", "LLM_EXTRA_PARAMS": "",
           "LLM_PROMPT_VERSION": ""}
    out = subprocess.check_output(
        [sys.executable, "-c", "from src import config as c; print(c.LLM_BASE_URL, c.LLM_MODEL, c.LLM_EXTRA_PARAMS, "
         "c.LLM_PROMPT_VERSION, "
         "c.LLM_TEMPERATURE, c.LLM_SEED)"], env=env, cwd=config.ROOT, text=True)
    assert out.split() == ["http://localhost:11434/v1", "llama3.2:3b", "{}", "v1", "0", "42"]


def test_llm_prompt_version_comes_from_env():
    out = subprocess.check_output([sys.executable, "-c", "from src import config as c; print(c.LLM_PROMPT_VERSION)"],
                                  env={**os.environ, "LLM_PROMPT_VERSION": "v2"}, cwd=config.ROOT, text=True)
    assert out.strip() == "v2"


def test_rag_uses_llm_prompt_version_unless_overridden(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROMPT_VERSION", "v2")
    assert RAG(FakeStore([]), FakeReranker([]), None).prompt.system == SYSTEM_PROMPT_V2
    assert RAG(FakeStore([]), FakeReranker([]), None, prompt_version="v1").prompt.system == SYSTEM_PROMPT


@pytest.mark.parametrize("args, version", [([], "v2"), (["--prompt-version", "v1"], "v1")])
def test_smoke_eval_prompt_version_flag_overrides_the_setting(monkeypatch, capsys, args, version):
    from evals import run_smoke
    made = []

    class FakeRAG:
        def __init__(self, generate, prompt_version):
            made.append(prompt_version)

        def ask(self, question):
            return {"answer": "x", "refused": True, "sources": [], "unverified_citations": []}

    monkeypatch.setattr(config, "LLM_PROMPT_VERSION", "v2")
    monkeypatch.setattr("src.rag.RAG", FakeRAG)
    run_smoke.main(args)
    assert made == [version] and f"PROMPT_VERSION: {version}\n" in capsys.readouterr().out


def test_smoke_eval_prints_model_and_prompt_version(monkeypatch, capsys):
    from evals import run_smoke
    made = {}

    class FakeRAG:
        def __init__(self, generate, prompt_version):
            made.update(generate=generate, version=prompt_version)

        def ask(self, question):
            return {"answer": "x", "refused": True, "sources": [], "unverified_citations": []}

    monkeypatch.setattr("src.rag.RAG", FakeRAG)
    run_smoke.main(["--model", "some/model", "--base-url", "http://host/v1", "--prompt-version", "v1"])
    out = capsys.readouterr().out
    assert out.startswith("LLM_MODEL: some/model\nPROMPT_VERSION: v1\n")
    assert made["version"] == "v1" and made["generate"].__self__.client.url == "http://host/v1/chat/completions"
