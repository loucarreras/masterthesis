import requests
import json
import pandas as pd
import os

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "medgemma1.5"

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure"
}

class OllamaSNOMEDClassifier:

    def __init__(self, model=MODEL):
        self.model = model
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

    def classify(self, term, context, start=None, end=None, note_id=None):

        prompt = self.build_prompt(term, context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        text = response.json()["response"]

        try:
            result = json.loads(text)
        except:
            result = {
                "term": term,
                "category": "Unknown",
                "reason": "Unable to classify"
            }

        
        result.update({
        "start": start,
        "end": end,
        "note_id": note_id,
        })

        self.results.append(result)
    
        return result
    
    def classify_hierarchical(self, term, context, snomed, all_ancestors, gt_id, current_level = 1, parent_id = None, start=None, end=None, note_id=None, trace=None):
        
        # Initialize trace
        if trace is None:
            trace = []

        def finalize_result(result):
            result.update({
                "start": start,
                "end": end,
                "note_id": note_id,
                "trace":trace,
                "level_reached": current_level,
                "gt_id": gt_id
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
        
        label_to_id = {v: k for k, v in labels.items()}
        
        # Build prompt
        prompt = self.build_prompt(term, context, labels)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        text = response.json()["response"]

        try:
            result = json.loads(text)
        except:
            result = {
                "term": term,
                "category": "Unknown",
                "reason": "Unable to classify"
            }
            return finalize_result(result)

        chosen_label = result.get("category")

        if parent_label and chosen_label == parent_label:
            result["concept_id"] = parent_id
            trace.append({
                "level": current_level,
                "chosen_label": chosen_label,
                "chosen_id": parent_id,
                "valid": True,
                "gt_reached": (parent_id == gt_id),
                "candidate_count": len(labels),
                "stopped_at_parent": True,
            })

        chosen_id = label_to_id.get(chosen_label)
        result["concept_id"] = chosen_id

        # Validation
        is_valid = False

        if chosen_id:
            is_valid = (chosen_id ==gt_id) or (chosen_id in all_ancestors)

        trace.append({
            "level": current_level,
            "chosen_label": chosen_label,
            "chosen_id": chosen_id,
            "valid": is_valid,
            "gt_reached": (chosen_id == gt_id),
            "candidate_count": len(labels),
            "stopped_at_parent": False,
        })

        # Stop conditions
        if not chosen_id:
            return finalize_result(result)
        
        if not snomed.get_children(chosen_id):
            return finalize_result(result)
        
        if not is_valid:
            return finalize_result(result)

        # Recursion
        return self.classify_hierarchical(
            term=term,
            context=context,
            snomed=snomed,
            all_ancestors=all_ancestors,
            gt_id=gt_id,
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