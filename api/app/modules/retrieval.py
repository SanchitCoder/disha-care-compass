"""Similar-case retrieval: char n-gram TF-IDF over the training split (robust to Hinglish spelling)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import data_loader
from ..data_loader import Case, load_taxonomy
from .normalise import clean_variants


@dataclass
class Hit:
    case: Case
    sim: float


class Retriever:
    def __init__(self, cases: list[Case]):
        self.cases = cases
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.mat = self.vec.fit_transform([clean_variants(c.text)[0] for c in cases])
        self.emergency = load_taxonomy().emergency_label

    def top_k(self, text: str, k: int = 6) -> list[Hit]:
        sims = (self.mat @ self.vec.transform([clean_variants(text)[0]]).T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        hits = [Hit(self.cases[i], float(sims[i])) for i in order]
        if not any(h.case.care_type == self.emergency for h in hits):   # contrast: keep one emergency example if close enough
            for i in np.argsort(-sims):
                if self.cases[i].care_type == self.emergency:
                    if sims[i] > 0.2:
                        hits[-1] = Hit(self.cases[i], float(sims[i]))
                    break
        return hits


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(data_loader.load_split(data_loader.TRAIN_SPLIT))
