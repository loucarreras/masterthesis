# import json
# import re
# import pandas as pd

# from ontology.llm_backends import LLMBackend, OllamaBackend

# DEFAULT_BACKEND = OllamaBackend("qwen3:8b")  # CHANGE FOR DIFFERENT MODELS

# LEVEL1 = {
#     "404684003": "Clinical finding",
#     "71388002": "Procedure",
#     "123037004": "Body structure",
# }


# def _strip_parens(label: str) -> str:
#     return re.sub(r'\s*\(.*?\)\s*$', '', label).strip()


# class SNOMEDClassifier:

#     def __init__(self, backend: LLMBackend = DEFAULT_BACKEND):
#         self.backend = backend
#         self.results = []

#     # ------------------------------------------------------------------
#     # Candidate label resolution
#     # ------------------------------------------------------------------
#     def get_candidate_labels(self, snomed, candidate_ids):
#         """
#         Map candidate concept_ids -> a clean, human-readable label the LLM
#         can read and choose from.
#         """
#         labels = {}
#         for cid in candidate_ids:
#             label = snomed.get_label(cid)
#             if label:
#                 labels[cid] = _strip_parens(label)
#         return labels

#     def _match_label_to_id(self, chosen_label, labels):
#         """Map the LLM's free-text chosen category back to a concept_id."""
#         if not chosen_label:
#             return None

#         chosen_norm = chosen_label.strip().lower()

#         # 1) exact match
#         for cid, lbl in labels.items():
#             if lbl.strip().lower() == chosen_norm:
#                 return cid

#         # 2) substring match (handles minor LLM paraphrasing)
#         for cid, lbl in labels.items():
#             lbl_norm = lbl.strip().lower()
#             if chosen_norm in lbl_norm or lbl_norm in chosen_norm:
#                 return cid

#         return None

#     # ------------------------------------------------------------------
#     # Prompting
#     # ------------------------------------------------------------------
#     def build_prompt(self, term, context, labels):
#         label_text = "\n".join(f"- {lbl}" for lbl in labels.values())

#         return f"""
# You are a clinical terminology expert.

# Classify the term into the SINGLE BEST matching SNOMED CT class, using the clinical context provided.

# Context: "{context}"

# Term: "{term}"

# Choose ONLY one from:

# {label_text}

# Instructions:
# - Pick the most specific category that accurately describes the term in the given context.
# - Choose exactly one option from the list above, copied verbatim.

# Strictly return JSON format like this, without any additional text:

# {{
#     "term": "{term}",
#     "category": "best category",
#     "reason": "short explanation"
# }}
# """

#     def _call(self, prompt: str, term: str) -> dict:
#         try:
#             text = self.backend.generate(prompt)
#             result = json.loads(text)
#             return result
#         except Exception:
#             return {"term": term, "category": "Unknown", "reason": "Unable to classify"}

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------
#     def classify(self, term, context, snomed, candidates, gt_id=None, start=None, end=None, note_id=None):
#         """
#         term        : mention text extracted from the note
#         context     : surrounding text/sentence for disambiguation
#         snomed      : SNOMEDHierarchy instance, used to resolve candidate labels
#         candidates  : list of candidate concept_ids (e.g. union of lexical + dense search)
#         gt_id       : ground-truth concept_id, carried through for evaluation only
#                       (never shown to the LLM)
#         start, end, note_id : span metadata, carried through to the output
#         """
#         labels = self.get_candidate_labels(snomed, candidates)

#         if not labels:
#             result = {
#                 "term": term,
#                 "category": None,
#                 "concept_id": None,
#                 "reason": "No candidates available",
#             }
#         else:
#             prompt = self.build_prompt(term, context, labels)
#             raw = self._call(prompt, term)
#             chosen_label = raw.get("category", "")
#             concept_id = self._match_label_to_id(chosen_label, labels)

#             result = {
#                 "term": term,
#                 "category": chosen_label,
#                 "concept_id": concept_id,
#                 "reason": raw.get("reason", ""),
#             }

#         result.update({
#             "gt_id": gt_id,
#             "start": start,
#             "end": end,
#             "note_id": note_id,
#         })

#         self.results.append(result)
#         return result

#     # ------------------------------------------------------------------
#     # Persistence
#     # ------------------------------------------------------------------
#     def save_results(self, filepath="classification_results.json"):
#         with open(filepath, "w", encoding="utf-8") as f:
#             json.dump(self.results, f, indent=4, ensure_ascii=False)

#     def save_results_csv(self, filepath="classification_results.csv"):
#         df = pd.DataFrame(self.results)
#         df.to_csv(filepath, index=False, encoding="utf-8")



import json
import re
import pandas as pd

from ontology.llm_backends import LLMBackend, OllamaBackend

DEFAULT_BACKEND = OllamaBackend("qwen3:8b")  # CHANGE FOR DIFFERENT MODELS

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}


def _strip_parens(label: str) -> str:
    return re.sub(r'\s*\(.*?\)\s*$', '', label).strip()


class SNOMEDClassifier:

    def __init__(self, backend: LLMBackend = DEFAULT_BACKEND):
        self.backend = backend
        self.results = []

    # ------------------------------------------------------------------
    # Candidate label resolution
    # ------------------------------------------------------------------
    def get_candidate_labels(self, snomed, candidate_ids):
        """
        Map candidate concept_ids -> a clean, human-readable label the LLM
        can read and choose from.

        NOTE: this assumes `snomed` exposes a method that returns a single
        preferred label string for a concept_id (e.g. the FSN or PT from
        sct2_Description). Adjust the method name below if your
        SNOMEDHierarchy class calls it something else
        (e.g. get_fsn / get_pt / get_label).
        """
        labels = {}
        for cid in candidate_ids:
            label = snomed.get_label(cid)  # <-- ADAPT if needed
            if label:
                labels[cid] = _strip_parens(label)
        return labels

    def _match_label_to_id(self, chosen_label, labels):
        """Map the LLM's free-text chosen category back to a concept_id."""
        if not chosen_label:
            return None

        chosen_norm = chosen_label.strip().lower()

        # 1) exact match
        for cid, lbl in labels.items():
            if lbl.strip().lower() == chosen_norm:
                return cid

        # 2) substring match (handles minor LLM paraphrasing)
        for cid, lbl in labels.items():
            lbl_norm = lbl.strip().lower()
            if chosen_norm in lbl_norm or lbl_norm in chosen_norm:
                return cid

        return None

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def build_prompt(self, term, context, labels):
        label_text = "\n".join(f"- {lbl}" for lbl in labels.values())

        return f"""
You are a clinical terminology expert.

Classify the term into the SINGLE BEST matching SNOMED CT class, using the clinical context provided.

Context: "{context}"

Term: "{term}"

Choose ONLY one from:

{label_text}

Instructions:
- Pick the most specific category that still accurately describes the term.
- Choose exactly one option from the list above, copied verbatim.

Strictly return JSON format like this, without any additional text:

{{
    "term": "{term}",
    "category": "best category",
    "reason": "short explanation"
}}
"""

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Many local LLMs wrap JSON in ```json ... ``` fences or add stray
        text before/after the object. Strip fences, then fall back to
        grabbing the first {...} block.
        """
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _call(self, prompt: str, term: str) -> dict:
        try:
            raw_text = self.backend.generate(prompt)
        except Exception as e:
            print(f"[SNOMEDClassifier] backend.generate() raised for term={term!r}: {e!r}")
            return {"term": term, "category": "Unknown", "reason": f"Backend error: {e}"}

        # Some backends return (text, metadata) instead of a plain string.
        if isinstance(raw_text, tuple):
            raw_text = raw_text[0]

        if not isinstance(raw_text, str):
            print(f"[SNOMEDClassifier] Unexpected generate() return type for term={term!r}: {type(raw_text)} -> {raw_text!r}")
            return {"term": term, "category": "Unknown", "reason": f"Unexpected backend return type: {type(raw_text)}"}

        cleaned = self._extract_json(raw_text)
        try:
            result = json.loads(cleaned)
            return result
        except Exception as e:
            print(f"[SNOMEDClassifier] JSON parse failed for term={term!r}: {e!r}")
            print(f"  Raw model output was: {raw_text!r}")
            return {"term": term, "category": "Unknown", "reason": f"JSON parse error: {e}"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, term, context, snomed, candidates, gt_id=None, start=None, end=None, note_id=None):
        """
        term        : mention text extracted from the note
        context     : surrounding text/sentence for disambiguation
        snomed      : SNOMEDHierarchy instance, used to resolve candidate labels
        candidates  : list of candidate concept_ids (e.g. union of lexical + dense search)
        gt_id       : ground-truth concept_id, carried through for evaluation only
                      (never shown to the LLM)
        start, end, note_id : span metadata, carried through to the output
        """
        labels = self.get_candidate_labels(snomed, candidates)

        if not labels:
            result = {
                "term": term,
                "category": None,
                "concept_id": None,
                "reason": "No candidates available",
            }
        else:
            prompt = self.build_prompt(term, context, labels)
            raw = self._call(prompt, term)
            chosen_label = raw.get("category", "")
            concept_id = self._match_label_to_id(chosen_label, labels)

            result = {
                "term": term,
                "category": chosen_label,
                "concept_id": concept_id,
                "reason": raw.get("reason", ""),
            }

        result.update({
            "gt_id": gt_id,
            "start": start,
            "end": end,
            "note_id": note_id,
            "candidate_ids": candidates,
        })

        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, filepath="classification_results.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)

    def save_results_csv(self, filepath="classification_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False, encoding="utf-8")