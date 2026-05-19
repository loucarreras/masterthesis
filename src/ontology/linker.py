from ontology.snomed_loader import load_snomed_terms
from ontology.faiss_index import SNOMEDIndex
from ontology.reranker_ollama import OllamaReranker

class SNOMEDLinker:

    def __init__(self, desc_file):

        self.terms = load_snomed_terms(desc_file)

        self.index = SNOMEDIndex()
        self.index.build(self.terms)

        self.reranker = OllamaReranker()

    def link(self, term):

        candidates = self.index.search(term, top_k=5)

        result = self.reranker.rerank(term, candidates)

        best = candidates[result["best_rank"] - 1]

        return {
            "term": term,
            "snomed_id": best["conceptId"],
            "preferred_term": best["term"],
            "confidence": result["confidence"]
        }