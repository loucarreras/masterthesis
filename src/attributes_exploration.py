import pandas as pd
from ontology.snomed_loader import SNOMEDHierarchy
import matplotlib.pyplot as plt
from scipy.stats import entropy

if __name__ == "__main__":

    # LOAD SNOMED HIERARCHY
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt",
        relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt",
        description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
    )
    print("SNOMED CT hierarchy loaded.")

    # LOADING DATA
    print("Loading ground truth annotations...")
    annotations_path = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
    annotations_df = pd.read_csv(annotations_path, dtype={"concept_id": str})
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    # LOADING PATH DATA
    print("Loading path characterization data...")
    path_characterization_path = "characterization_results.csv"
    path_df = pd.read_csv(path_characterization_path, dtype={"concept_id": str})
    print(f"Loaded {len(path_df)} concepts.")

    rows = []
    seen_ancestors = set()
    for row in path_df.itertuples():

        concept_id = row.concept_id
        ancestors = snomed.get_all_ancestors(concept_id)
        ancestors.add(concept_id)

        for ancestor in ancestors:

            if ancestor in seen_ancestors:
                continue

            seen_ancestors.add(ancestor)

            attributes = snomed.get_attributes(ancestor)
            num_attributes = len(attributes)
            
            rows.append(
                {
                    "concept_id": ancestor,
                    "label": snomed.get_label(ancestor),
                    "attributes": attributes,
                    "num_attributes": num_attributes,
                    "attribute_types": [attr[0] for attr in attributes],
                    "attribute_types_labels": [snomed.get_label(attr[0]) for attr in attributes]
                }
            )

    characterization_df = pd.DataFrame(rows)
    characterization_df.to_csv("attributes_ancestors_characterization.csv", index=False, encoding="utf-8")

    print(characterization_df.head(10))
    print(characterization_df.describe(include="all").transpose())
    print("Number of concepts with attributes:", (characterization_df["num_attributes"] > 0).sum())
    print("Number of unique attributes:", characterization_df["attributes"].explode().nunique())
    print("Most common attributes:", characterization_df["attributes"].explode().value_counts().head(11))
    print("Number of unique attribute types:", characterization_df["attribute_types"].explode().nunique())
    print("Most common attribute types:", characterization_df["attribute_types_labels"].explode().value_counts().head(10))

    attr_counts = (
    characterization_df["attributes"]
    .explode()
    .dropna()
    .value_counts()
    .head(11)
)

    for (attr_type_id, target_id), count in attr_counts.items():

        attr_label = snomed.get_label(attr_type_id)
        target_label = snomed.get_label(target_id)

        print(f"{attr_label} -> {target_label}: {count}")

    # Count attribute types
    attr_counts = (
        characterization_df["attribute_types_labels"]
        .explode()
        .value_counts()
        .head(11)
    )

    # Plot
    plt.figure(figsize=(12, 6))

    attr_counts.sort_values().plot(kind="barh")

    plt.title("Most Common SNOMED CT Attribute Types")
    plt.xlabel("Attribute Type")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()