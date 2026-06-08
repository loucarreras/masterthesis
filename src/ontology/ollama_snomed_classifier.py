import requests
import json
import pandas as pd
import os
from ontology.llm_backends import LLMBackend, OllamaBackend

DEFAULT_BACKEND = OllamaBackend("medgemma1.5") # CHANGE FOR DIFFERENT MODELS

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure"
}

def _strip_parens(label:str) -> str:
    return re.sub(r'\s*\(.*?\)\s*$', '', label).strip()

class OllamaSNOMEDClassifier:

    def __init__(self, backend: LLMBackend = DEFAULT_BACKEND):
        self.backend = backend
        self.results = []

    def build_prompt(self, term, context, labels, parent_label=None):
        
        all_options = dict(labels)
        if parent_label:
            all_options["__parent__"] = parent_label
 
        label_text = "\n".join([f"{v}" for v in all_options.values()])
 
        parent_instruction = (
            f'\n- If "{parent_label}" is already the most precise match, choose it and do not go deeper.'
            if parent_label else ""
        )
 
        return f"""
You are a clinical terminology expert.
 
Classify the term into the SINGLE BEST matching SNOMED CT top-level hierarchy, using the clinical context provided.
 
Context: "{context}"
 
Term: "{term}"
 
Choose ONLY one from:
 
{label_text}
 
Instructions:
- Pick the most specific category that still accurately describes the term.{parent_instruction}
 
Strictly return JSON format like this, without any additional text:
 
{{
    "term": "{term}",
    "category": "best category",
    "reason": "short explanation"
}}
"""
    def _call(self, prompt: str, term: str) -> dict:

        try: 
            text = self.backend.generate(prompt)
            result = json.loads(text)
            return result
        except Exception:
            return {"term": term, "category": "Unknown", "reason": "Unable to classify"}
    

    def classify(self, term, context, start=None, end=None, note_id=None):

        prompt = self.build_prompt(term, context)
        result = self._call(prompt, term)

        result.update({
        "start": start,
        "end": end,
        "note_id": note_id,
        })

        self.results.append(result)
    
        return result
    
    def classify_hierarchical(self, term, context, snomed, current_level = 1, parent_id = None, start=None, end=None, note_id=None, trace=None):
        
        # Initialize trace
        if trace is None:
            trace = []

        def finalize_result(result):
            result.update({
                "start": start,
                "end": end,
                "note_id": note_id,
                "trace":trace,
                "level_reached": current_level
                })

            self.results.append(result)

            return result

        # Root level
        if current_level == 1:
            labels = LEVEL1
            parent_label = None
        else:
            children_ids = snomed.get_children(parent_id)
            labels = {cid: snomed.get_label(cid) for cid in children_ids}
            parent_label = snomed.get_label(parent_id)

        # Leaf node
        if not labels:
            return finalize_result({
                "term": term,
                "category": snomed.get_label(parent_id),
                "reason": "Leaf node reached",
                "concept_id": parent_id,
            })
        
        label_to_id = {_strip_parens(v): k for k, v in labels.items()}
        
        # Build prompt
        prompt = self.build_prompt(term, context, labels)
        result = self._call(prompt, term)

        if result.get("category") == "Unknown":
            return finalize_result(result)
        
        chosen_label = _strip_parens(result.get("category"))

        if parent_label and _strip_parens(chosen_label) == _strip_parens(parent_label):
            result["concept_id"] = parent_id
            trace.append({
                "level": current_level,
                "chosen_label": chosen_label,
                "chosen_id": parent_id,
                "candidate_count": len(labels),
                "stopped_at_parent": True,
            })
            return finalize_result(result)

        chosen_id = label_to_id.get(chosen_label)
        result["concept_id"] = chosen_id


        trace.append({
            "level": current_level,
            "chosen_label": chosen_label,
            "chosen_id": chosen_id,
            "candidate_count": len(labels),
            "stopped_at_parent": False,
        })

        # Stop conditions
        if not chosen_id:
            return finalize_result(result)
        
        if not snomed.get_children(chosen_id):
            return finalize_result(result)
        
        # Recursion
        return self.classify_hierarchical(
            term=term,
            context=context,
            snomed=snomed,
            current_level=current_level+1,
            parent_id=chosen_id,
            start=start,
            end=end,
            note_id=note_id,
            trace=trace,
        )

    def save_results(self, filepath="classification_results.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
    
    def save_results_csv(self, filepath="classification_results.csv"):

        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False, encoding="utf-8")