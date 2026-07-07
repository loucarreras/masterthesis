"""
Combines the lexical (Prior-equivalent) and dense (BLINK-equivalent)
candidate sets into the final Ec = Ep u Er, capped at FINAL_TOP_K,
mirroring the paper's step 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from dense_index import DenseIndex
from lexical_index import EmpiricalPrior, LexicalIndex


@dataclass
class Candidate:
    concept_id: str
    lexical_score: float | None = None
    dense_score: float | None = None
    prior_score: float | None = None

    @property
    def sources(self) -> list[str]:
        s = []
        if self.lexical_score is not None:
            s.append("lexical")
        if self.dense_score is not None:
            s.append("dense")
        if self.prior_score is not None:
            s.append("prior")
        return s


class CandidateGenerator:
    def __init__(self, lexical_index: LexicalIndex, dense_index: DenseIndex, empirical_prior: EmpiricalPrior = None):
        self.lexical_index = lexical_index
        self.dense_index = dense_index
        self.empirical_prior = empirical_prior

    def generate(self, mention: str, top_k: int = config.FINAL_TOP_K) -> list[Candidate]:
        lexical_hits = dict(self.lexical_index.search(mention, top_n=config.BM25_TOP_N))
        dense_hits = dict(self.dense_index.search(mention, top_n=config.DENSE_TOP_N))
        prior_hits = dict(self.empirical_prior.lookup(mention)) if self.empirical_prior else {}

        all_ids = set(lexical_hits) | set(dense_hits) | set(prior_hits)
        candidates = [
            Candidate(
                concept_id=cid,
                lexical_score=lexical_hits.get(cid),
                dense_score=dense_hits.get(cid),
                prior_score=prior_hits.get(cid),
            )
            for cid in all_ids
        ]

        # Ranking: prior hits first (highest-precision signal when available),
        # then interleave by normalized lexical/dense score. This is a simple,
        # transparent default -- tune/replace with a learned re-ranker later.
        def sort_key(c: Candidate):
            has_prior = c.prior_score is not None
            best_secondary = max(c.lexical_score or 0, c.dense_score or 0)
            return (not has_prior, -(c.prior_score or 0), -best_secondary)

        candidates.sort(key=sort_key)
        return candidates[:top_k]