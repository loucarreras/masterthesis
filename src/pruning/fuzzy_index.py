"""
Fuzzy-matching candidate generator using RapidFuzz (C++-backed edit-distance
scoring -- far faster than pure-Python Levenshtein at this scale).

Matches the same interface as LexicalIndex (BM25), so it's a drop-in
replacement anywhere LexicalIndex is used:

    .search(mention, top_n) -> list[(concept_id, score)]
    .search_verbose(mention, top_n) -> list[dict] with matched_term

Scorer choice matters a lot here -- rapidfuzz.fuzz offers several:
  - fuzz.ratio            plain normalized Levenshtein similarity (0-100).
                           Sensitive to length differences and word order.
  - fuzz.token_sort_ratio  same, but tokenizes + sorts words first, so word
                           order doesn't matter ("pain chest" ~= "chest pain").
  - fuzz.token_set_ratio   like token_sort_ratio, but also ignores duplicate/
                           extra words -- good when the mention or SNOMED
                           term has extra qualifiers ("acute chest pain" vs
                           "chest pain").
  - fuzz.WRatio (default)  a weighted combination of the above, generally the
                           most robust default for messy real-world text.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process
BM25_TOP_N = 20

class FuzzyIndex:
    def __init__(self, synonyms: dict[str, list[str]], scorer=fuzz.WRatio):
        self.doc_concept_ids: list[str] = []
        self.doc_terms: list[str] = []
        for concept_id, terms in synonyms.items():
            for term in terms:
                self.doc_concept_ids.append(concept_id)
                self.doc_terms.append(term)
        self.scorer = scorer

    def search(self, mention: str, top_n: int = BM25_TOP_N) -> list[tuple[str, float]]:
        # over-fetch raw string matches since multiple synonyms map to the same concept
        raw_matches = process.extract(
            mention, self.doc_terms, scorer=self.scorer, limit=top_n * 5
        )
        best_per_concept: dict[str, float] = {}
        for _term, score, idx in raw_matches:
            concept_id = self.doc_concept_ids[idx]
            if score > best_per_concept.get(concept_id, -1):
                best_per_concept[concept_id] = score
        ranked = sorted(best_per_concept.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

class BlockedFuzzyIndex:
    """Same interface as FuzzyIndex, but scales to the full SNOMED corpus.

    Exhaustive fuzzy scoring against ~1.4M description strings is too slow
    to run for every mention in a real evaluation set (extrapolates to
    hours, even single-mention). This class fixes that with blocking:
    a cheap character-trigram overlap prefilter narrows each mention down
    to a small shortlist of plausible terms FIRST, then exact RapidFuzz
    scoring only runs on that shortlist.

    Trigrams (not whole words) are used for blocking specifically so typos
    still work: a 1-2 character misspelling still shares most trigrams with
    the correct spelling, whereas word-overlap blocking would require an
    exact matching token and reintroduce BM25's blind spot.

    Trade-off: if a mention shares almost no trigrams with the true term at
    all (e.g. a heavily contracted abbreviation like "SOB"), it can still be
    missed by blocking even though full RapidFuzz scoring might have found
    it via WRatio's partial/token logic. In practice this is rare for
    genuine typos/near-misspellings, which is the main use case for adding
    fuzzy matching in the first place.
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]],
        scorer=fuzz.WRatio,
        n: int = 3,
        shortlist_size: int = 500,
    ):
        self.doc_concept_ids: list[str] = []
        self.doc_terms: list[str] = []
        self.scorer = scorer
        self.n = n
        self.shortlist_size = shortlist_size

        trigram_index: dict[str, list[int]] = {}
        for concept_id, terms in synonyms.items():
            for term in terms:
                doc_idx = len(self.doc_terms)
                self.doc_concept_ids.append(concept_id)
                self.doc_terms.append(term)
                for trigram in self._trigrams(term):
                    trigram_index.setdefault(trigram, []).append(doc_idx)
        self.trigram_index = trigram_index

    def _trigrams(self, text: str) -> set[str]:
        text = f"  {text.lower()}  "  # pad so short words still produce trigrams
        return {text[i : i + self.n] for i in range(len(text) - self.n + 1)}

    def _shortlist(self, mention: str) -> list[int]:
        from collections import Counter

        overlap_counts: Counter = Counter()
        for trigram in self._trigrams(mention):
            for doc_idx in self.trigram_index.get(trigram, ()):
                overlap_counts[doc_idx] += 1
        return [idx for idx, _ in overlap_counts.most_common(self.shortlist_size)]

    def search(self, mention: str, top_n: int = BM25_TOP_N) -> list[tuple[str, float]]:
        shortlist_idx = self._shortlist(mention)
        if not shortlist_idx:
            return []
        shortlist_terms = [self.doc_terms[i] for i in shortlist_idx]
        raw_matches = process.extract(mention, shortlist_terms, scorer=self.scorer, limit=top_n * 5)

        best_per_concept: dict[str, float] = {}
        for _term, score, local_idx in raw_matches:
            concept_id = self.doc_concept_ids[shortlist_idx[local_idx]]
            if score > best_per_concept.get(concept_id, -1):
                best_per_concept[concept_id] = score
        ranked = sorted(best_per_concept.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def search_verbose(self, mention: str, top_n: int = BM25_TOP_N) -> list[dict]:
        shortlist_idx = self._shortlist(mention)
        if not shortlist_idx:
            return []
        shortlist_terms = [self.doc_terms[i] for i in shortlist_idx]
        raw_matches = process.extract(mention, shortlist_terms, scorer=self.scorer, limit=top_n * 5)

        best_per_concept: dict[str, tuple[float, str]] = {}
        for term, score, local_idx in raw_matches:
            concept_id = self.doc_concept_ids[shortlist_idx[local_idx]]
            if score > best_per_concept.get(concept_id, (-1, ""))[0]:
                best_per_concept[concept_id] = (score, term)
        ranked = sorted(best_per_concept.items(), key=lambda x: x[1][0], reverse=True)
        return [
            {"concept_id": cid, "score": score, "matched_term": term}
            for cid, (score, term) in ranked[:top_n]
        ]