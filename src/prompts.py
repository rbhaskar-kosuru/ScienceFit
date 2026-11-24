"""Versioned prompts. Plain system/user text only, so any generator can use them.
config.LLM_PROMPT_VERSION picks one; the smoke eval can override it with --prompt-version."""
from typing import Callable, NamedTuple

SYSTEM_PROMPT = (
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


def build_prompt(question: str, hits: list[dict]) -> str:
    excerpts = "\n\n".join(format_excerpt(h) for h in hits)
    ids = ", ".join(dict.fromkeys(h["paper"] for h in hits))
    return (f"{excerpts}\n\nQuestion: {question}\n\n"
            f"Allowed citation ids: {ids}\n"
            "Answer in your own words. Cite only with an allowed id in the format [id]; "
            "don't reproduce the excerpt markup.")


def format_excerpt(h: dict) -> str:
    """<excerpt id="carvalho_2022" section="introduction"> text </excerpt>. No title/journal/design, so there's
    no header for the model to copy; unknown sections get no section attribute."""
    section = h.get("section")
    attr = f' section="{section}"' if section and section != "unknown" else ""
    return f'<excerpt id="{h["paper"]}"{attr}>\n{h["text"]}\n</excerpt>'


SYSTEM_PROMPT_V2 = (
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


def build_prompt_v2(question: str, hits: list[dict]) -> str:
    excerpts = "\n\n".join(format_excerpt(h) for h in hits)
    ids = ", ".join(dict.fromkeys(h["paper"] for h in hits))
    return (f"{excerpts}\n\nQuestion: {question}\n\n"
            f"Allowed ids: {ids}\n"
            "Answer the question, following the rules. Write every citation as [id].")


class Prompt(NamedTuple):
    system: str
    build: Callable[[str, list[dict]], str]


PROMPTS = {
    "v1": Prompt(SYSTEM_PROMPT, build_prompt),
    "v2": Prompt(SYSTEM_PROMPT_V2, build_prompt_v2),
}


def get_prompt(version: str) -> Prompt:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version {version!r}; available: {', '.join(PROMPTS)}")
    return PROMPTS[version]
