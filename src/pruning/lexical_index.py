from __future__ import annotations

import re
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi

BM25_TOP_N = 20

def _tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class LexicalIndex:
    """BM25 index over every (conceptId, term) pair. Each description string
    is a separate BM25 'document'; results are aggregated to concept level."""

    def __init__(self, synonyms: dict[str, list[str]]):
        self.doc_concept_ids: list[str] = []
        self.doc_terms: list[str] = []
        tokenized_corpus = []
        for concept_id, terms in synonyms.items():
            for term in terms:
                self.doc_concept_ids.append(concept_id)
                self.doc_terms.append(term)
                tokenized_corpus.append(_tokenize(term))
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, mention: str, top_n: int = BM25_TOP_N) -> list[tuple[str, float]]:
        """Returns up to top_n (conceptId, score) pairs, deduped by concept,
        keeping each concept's best-scoring description match."""
        query_tokens = _tokenize(mention)
        scores = self.bm25.get_scores(query_tokens)

        best_per_concept: dict[str, float] = {}
        for concept_id, score in zip(self.doc_concept_ids, scores):
            if score <= 0:
                continue
            if score > best_per_concept.get(concept_id, -1):
                best_per_concept[concept_id] = score

        ranked = sorted(best_per_concept.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    
    def search_verbose(self, mention: str, top_n: int = BM25_TOP_N) -> list[dict]:
        """Returns up to top_n (conceptId, score, term) tuples, but also which specific synonym string
        produced the best match for each concept."""

        query_tokens = _tokenize(mention)
        scores = self.bm25.get_scores(query_tokens)

        best_per_concept: dict[str, tuple[float, str]] = {}
        for concept_id, score, term in zip(self.doc_concept_ids, scores, self.doc_terms):
            if score <= 0:
                continue
            if score > best_per_concept.get(concept_id, (-1, ""))[0]:
                best_per_concept[concept_id] = (score, term)

        ranked = sorted(best_per_concept.items(), key=lambda x: x[1][0], reverse=True)
        return [
            {"concept_id": concept_id, "score": score, "matched_term": term}
            for concept_id, (score, term) in ranked[:top_n]
        ]


class EmpiricalPrior:
    """Optional: mention-string -> concept_id frequency counts, mined from
    a training split of gold annotations (analogous to the hyperlink Prior,
    but corpus-derived instead of Wikipedia-derived).

    Expects an iterable of (mention_text, gold_concept_id) pairs from your
    training annotations.
    """

    def __init__(self, mention_gold_pairs: list[tuple[str, str]]):
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for mention, concept_id in mention_gold_pairs:
            key = mention.lower().strip()
            self.counts[key][concept_id] += 1

    def lookup(self, mention: str, top_n: int = 5) -> list[tuple[str, float]]:
        key = mention.lower().strip()
        counter = self.counts.get(key)
        if not counter:
            return []
        total = sum(counter.values())
        return [(cid, n / total) for cid, n in counter.most_common(top_n)]