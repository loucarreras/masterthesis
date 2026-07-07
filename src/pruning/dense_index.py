"""
Dense candidate generator: the domain replacement for BLINK.

SapBERT (Liu et al., 2021) is trained with self-alignment contrastive
learning directly on UMLS synonym pairs (SNOMED CT is a UMLS source
vocabulary), so mention embeddings and concept-name embeddings live in a
shared space suited to nearest-neighbour retrieval -- the same role BLINK
plays for Wikipedia entities.

We embed every synonym string once (cached to disk), build a FAISS index,
and at query time embed the mention and retrieve nearest concept names,
then aggregate to concept level (best-matching synonym per concept).
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import config

logger = logging.getLogger(__name__)


class SapBERTEncoder:
    def __init__(self, model_name: str = config.SAPBERT_MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch, padding=True, truncation=True, max_length=25, return_tensors="pt"
            ).to(self.device)
            out = self.model(**toks)
            # SapBERT convention: use the [CLS] token embedding
            cls_vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_vecs.append(cls_vecs)
        vecs = np.concatenate(all_vecs, axis=0).astype("float32")
        faiss.normalize_L2(vecs)  # so inner product == cosine similarity
        return vecs


class DenseIndex:
    def __init__(self, synonyms: dict[str, list[str]], encoder: SapBERTEncoder = None, cache_dir: Path = None):
        self.encoder = encoder or SapBERTEncoder()
        cache_dir = cache_dir or config.CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_cache = cache_dir / "sapbert_embeddings.npy"
        ids_cache = cache_dir / "sapbert_concept_ids.npy"
        terms_cache = cache_dir / "sapbert_terms.npy"

        self.doc_concept_ids: list[str] = []
        self.doc_terms: list[str] = []
        for concept_id, terms in synonyms.items():
            for term in terms:
                self.doc_concept_ids.append(concept_id)
                self.doc_terms.append(term)

        if emb_cache.exists() and ids_cache.exists():
            logger.info("Loading cached SapBERT embeddings from %s", cache_dir)
            embeddings = np.load(emb_cache)
            cached_ids = np.load(ids_cache, allow_pickle=True).tolist()
            assert cached_ids == self.doc_concept_ids, "Cache mismatch, delete cache and rebuild."
        else:
            logger.info("Encoding %d synonym strings with SapBERT (one-off, cached afterwards)", len(self.doc_terms))
            embeddings = self.encoder.encode(self.doc_terms)
            np.save(emb_cache, embeddings)
            np.save(ids_cache, np.array(self.doc_concept_ids, dtype=object))
            np.save(terms_cache, np.array(self.doc_terms, dtype=object))

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, mention: str, top_n: int = config.DENSE_TOP_N) -> list[tuple[str, float]]:
        query_vec = self.encoder.encode([mention])
        # over-fetch raw nearest neighbours since multiple synonyms map to the same concept
        raw_k = min(top_n * 5, self.index.ntotal)
        scores, indices = self.index.search(query_vec, raw_k)

        best_per_concept: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            concept_id = self.doc_concept_ids[idx]
            if score > best_per_concept.get(concept_id, -1):
                best_per_concept[concept_id] = float(score)

        ranked = sorted(best_per_concept.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]