import pandas as pd
from collections import defaultdict

# FILE PATHS
ANNOTATION_FILE = "snomed-ct-entity-linking-challenge-1.2.1/train_annotations.csv"
REL_FILE = "SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z/Snapshot/Terminology/sct2_Relationship_Snapshot_INT_20260401.txt"

# LOAD ANNOTATIONS
ann = pd.read_csv(ANNOTATION_FILE, dtype={"concept_id": str})

print(f"Loaded {len(ann)} annotations.")

# Ensure integer concept ids
ann["concept_id"] = (
    ann["concept_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)

# LOAD SNOMED RELATIONSHIPS
rel = pd.read_csv(REL_FILE, sep="\t", dtype=str)

# Keep active IS-A relationships
isa = rel[
    (rel["active"] == "1") &
    (rel["typeId"] == "116680003")
]

# child -> parents graph
parents = defaultdict(list)

for _, row in isa.iterrows():
    child = int(row["sourceId"])
    parent = int(row["destinationId"])
    parents[child].append(parent)

# TOP LEVEL HIERARCHIES
TOP_LEVEL = {
    404684003: "Clinical finding",
    71388002: "Procedure",
    123037004: "Body structure",
    105590001: "Substance",
    373873005: "Pharmaceutical / biologic product",
    260787004: "Physical object",
    272379006: "Event",
    243796009: "Situation with explicit context",
    362981000: "Qualifier value",
    419891008: "Record artifact"
}

# RECURSIVE ANCESTOR SEARCH
def get_top_levels(code, visited=None):
    if visited is None:
        visited = set()

    if code in TOP_LEVEL:
        return {(code, TOP_LEVEL[code])}

    if code in visited:
        return set()

    visited.add(code)

    results = set()

    for p in parents.get(code, []):
        results |= get_top_levels(p, visited)

    return results

# MAP ALL CONCEPTS
def map_code_and_ids(code):
    vals = get_top_levels(code)

    if len(vals) == 0:
        return "Unknown", ""
    
    labels = sorted([v[1] for v in vals])
    codes = sorted([str(v[0]) for v in vals])

    return "; ".join(labels), "; ".join(codes)

ann[["top_level_class", "top_level_code"]] = ann["concept_id"].apply(
    lambda x: pd.Series(map_code_and_ids(int(x)))
)
# SAVE
ann.to_csv("train_annotations_with_top_class.csv", index=False)

print("Saved: train_annotations_with_top_class.csv")
print(ann[["concept_id", "span", "top_level_class"]].head(10))