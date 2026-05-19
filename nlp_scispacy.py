import pandas as pd
import spacy
from pathlib import Path

# -----------------------------
# 1) Load notes
# -----------------------------
paths = [
    r"C:\DESTINATION\note\discharge.csv.gz",
    # r"C:\DESTINATION\note\radiology.csv.gz",
]

dfs = [pd.read_csv(p, compression="gzip", low_memory=False) for p in paths]
notes_df = pd.concat(dfs, ignore_index=True)

if "text" in notes_df.columns:
    notes_texts = notes_df["text"].fillna("").astype(str).tolist()
else:
    notes_texts = notes_df.fillna("").astype(str).agg(" ".join, axis=1).tolist()

print(f"Loaded {len(notes_texts)} notes.")

clinical_notes = notes_texts[:2]

# -----------------------------
# 2) Load NLP pipeline
# -----------------------------
nlp = spacy.load("en_core_sci_sm")

if "scispacy_linker" not in nlp.pipe_names:
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "linker_name": "umls",
            "resolve_abbreviations": True,
            "k": 5,
            "threshold": 0.75,
            "max_entities_per_mention": 10,
        },
    )

linker = nlp.get_pipe("scispacy_linker")

# Optional: if your model is weak at sentence segmentation
# nlp.max_length = 2_000_000

# -----------------------------
# 3) Helpers
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    rows = []

    for ent in doc.ents:
        linked = []
        for cui, score in ent._.kb_ents[:5]:
            kb_ent = linker.kb.cui_to_entity.get(cui)
            linked.append(
                {
                    "cui": cui,
                    "score": float(score),
                    "name": kb_ent.canonical_name if kb_ent else None,
                }
            )

        rows.append(
            {
                "mention": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "links": linked,
            }
        )

    return rows

def keep_best_link(entity_row):
    if not entity_row["links"]:
        return None
    return max(entity_row["links"], key=lambda x: x["score"])

# -----------------------------
# 4) Run extraction
# -----------------------------
all_results = []

for i, text in enumerate(clinical_notes):
    ents = extract_entities(text)
    for e in ents:
        best = keep_best_link(e)
        all_results.append(
            {
                "note_id": i,
                "mention": e["mention"],
                "label": e["label"],
                "start": e["start"],
                "end": e["end"],
                "best_cui": best["cui"] if best else None,
                "best_score": best["score"] if best else None,
                "best_name": best["name"] if best else None,
            }
        )

results_df = pd.DataFrame(all_results)
print(results_df.head(20))

# ------------------------
# 5) Second Mapping layer: SNOMED post-processing
# ------------------------

snomed_map = pd.read_csv("cui_to_snomed.csv")

final_df = results_df.merge(snomed_map, left_on="best_cui", right_on="cui", how="left")
final_df["is_snomed_mapped"] = final_df["snomed_id"].notna()

print(final_df[["note_id", "mention", "best_cui", "snomed_id", "snomed_term"]].head(20))
