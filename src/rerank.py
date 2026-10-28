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
        """Relevance in 0-1 per passage (sigmoid of the cross-encoder logit)."""
        if not passages:
            return []
        batch = self.tokenizer(
            [question] * len(passages), passages,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        return torch.sigmoid(self.model(**batch).logits.view(-1)).tolist()
