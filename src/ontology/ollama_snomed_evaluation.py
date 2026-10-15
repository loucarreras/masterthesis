import requests
import json
import pandas as pd
import os
import re
from ontology.llm_backends import LLMBackend, OllamaBackend
import traceback

DEFAULT_BACKEND = OllamaBackend("gemma4:26b", think=False, num_predict=2048) # CHANGE FOR DIFFERENT MODELS

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure"
}

def _strip_parens(label:str) -> str:
    return re.sub(r'\s*\(.*?\)\s*$', '', label).strip()

class SNOMEDClassifier:

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

    def _call(self, prompt: str, term: str):
        text = None
        try:
            text, total_time = self.backend.generate(prompt)
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("no JSON object found in model output")
            return json.loads(match.group(0)), total_time
        except Exception as e:
            print(f"[LLM ERROR] term={term!r}: {type(e).__name__}: {e}")
            print(f"[RAW OUTPUT] {text!r}")
            return {"term": term, "category": "Unknown", "reason": "Unable to classify"}, None
        
    def classify_hierarchical(self, term, context, snomed, all_ancestors, gt_id, current_level = 1,
                               parent_id = None, start=None, end=None, note_id=None, trace=None,
                               child_map=None, roots=None):
        """
        child_map / roots: optional pruned-subtree navigation, as produced by
        subtree_pruner.build_allowed_subtree() / get_roots(). When both are
        provided, the classifier navigates ONLY this pruned subtree:
          - level 1 offers the subtree's roots instead of the fixed LEVEL1
            top-level categories
          - each subsequent level offers child_map[parent_id] instead of
            snomed.get_children(parent_id)
        When omitted (the default), behavior is unchanged from before --
        full-hierarchy navigation starting from LEVEL1.
        """

        # Initialize trace
        if trace is None:
            trace = []

        using_subtree = child_map is not None

        def finalize_result(result):
            result.update({
                "start": start,
                "end": end,
                "note_id": note_id,
                "trace":trace,
                "level_reached": current_level,
                "gt_id": gt_id,
                "context": context,
            })

            self.results.append(result)

            return result

        # Root level
        if current_level == 1:
            if using_subtree:
                labels = {rid: snomed.get_label(rid) for rid in roots}
            else:
                labels = LEVEL1
            parent_label = None
        else:
            if using_subtree:
                children_ids = child_map.get(parent_id, [])
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
        result, total_time = self._call(prompt, term)

        if result.get("category") == "Unknown":
            return finalize_result(result)
        
        chosen_label = _strip_parens(result.get("category"))

        if parent_label and _strip_parens(chosen_label) == _strip_parens(parent_label):
            result["concept_id"] = parent_id
            trace.append({
                "level": current_level,
                "chosen_label": chosen_label,
                "chosen_id": parent_id,
                "valid": True,
                "gt_reached": (parent_id == gt_id),
                "candidate_count": len(labels),
                "stopped_at_parent": True,
                "call_time": total_time,
            })
            return finalize_result(result)

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
            "call_time": total_time
        })

        # Stop conditions
        if not chosen_id:
            return finalize_result(result)

        if using_subtree:
            has_children = bool(child_map.get(chosen_id))
        else:
            has_children = bool(snomed.get_children(chosen_id))

        if not has_children:
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
            child_map=child_map,
            roots=roots,
        )

    def save_results(self, filepath="classification_results.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
    
    def save_results_csv(self, filepath="classification_results.csv"):

        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False, encoding="utf-8")