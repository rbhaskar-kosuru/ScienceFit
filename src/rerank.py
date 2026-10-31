"""Cross-encoder reranker: reads question and chunk together, so it ranks relevance more sharply than vector similarity."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    @torch.no_grad()
    def score(self, question: str, passages: list[str]) -> list[float]:
        """Raw cross-encoder logit per passage (about -11 to +11). Callers take the sigmoid for a 0-1 relevance."""
        if not passages:
            return []
        batch = self.tokenizer(
            [question] * len(passages), passages,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        return self.model(**batch).logits.view(-1).tolist()
