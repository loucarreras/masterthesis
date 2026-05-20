import pandas as pd
from ontology.snomed_loader import SNOMEDHierarchy
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if os.path.exists("characterization_results.csv"):
        
        print("Loading characterization results...")
        characterization_df = pd.read_csv("characterization_results.csv", dtype={"concept_id": str})
        print(f"Loaded {len(characterization_df)} characterization results.")

    else:
        # LOADING DATA
        print("Loading ground truth annotations...")
        annotations_path = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
        annotations_df = pd.read_csv(annotations_path, dtype={"concept_id": str})
        print(f"Loaded {len(annotations_df)} ground truth annotations.")

        # LOAD SNOMED HIERARCHY
        print("Loading SNOMED CT hierarchy...")
        snomed = SNOMEDHierarchy(
            concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260401.txt",
            relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260401.txt",
            description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260401.txt"
        )
        print("SNOMED CT hierarchy loaded.")

        rows = []
        for concept_id in annotations_df["concept_id"].unique():

            paths = snomed.get_all_ancestor_paths(concept_id)
            depths = snomed.get_depths(concept_id)
            
            rows.append(
                {
                    "concept_id": concept_id,
                    "label": snomed.get_label(concept_id),
                    "is_leaf": snomed.is_leaf(concept_id),
                    "is_polyhierarchical": snomed.is_polyhierarchical(concept_id),
                    "num_paths": len(paths),
                    "min_depth": depths["min_depth"],
                    "max_depth": depths["max_depth"],
                    "all_depths": depths["all_depths"],
                    "num_parents": len(snomed.get_parents(concept_id)),
                    "num_children": len(snomed.get_children(concept_id)),
                    "all_paths": paths,
                    "num_ancestors": len(snomed.get_all_ancestors(concept_id)),
                    "num_descendants": len(snomed.get_all_descendants(concept_id))
                }
            )
        
        characterization_df = pd.DataFrame(rows)
        characterization_df.to_csv("characterization_results.csv", index=False, encoding="utf-8")
    
    print("Characterization summary:")
    print(characterization_df.head(10))
    print(characterization_df.describe(include="all").transpose())
    summary = {
        "num_annotations": len(characterization_df),
        "num_unique_concepts": characterization_df["concept_id"].nunique(),
        "num_polyhierarchical": characterization_df["is_polyhierarchical"].sum(),
        "num_leaves": characterization_df["is_leaf"].sum(),
        "avg_num_paths": characterization_df["num_paths"].mean(),
    }
    
    print("Distribution of depths:")
    print(characterization_df["min_depth"].describe())
    print(characterization_df["max_depth"].describe())

    print("Distribution of polyhierarchical concepts:")
    poly_df = characterization_df[characterization_df["is_polyhierarchical"]]
    summary["polyhierarchical_ratio"] = len(poly_df) / len(characterization_df)
    summary["avg_num_paths"] = characterization_df["num_paths"].mean()
    summary["max_num_paths"] = characterization_df["num_paths"].max()
    print(characterization_df["num_paths"].describe())


    print("Summary statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    max_paths = characterization_df["num_paths"].max()

    max_path_nodes = characterization_df[characterization_df["num_paths"] == max_paths]

    print(max_path_nodes[
        ["concept_id", "label", "num_paths"]
    ])

    max_children = characterization_df["num_children"].max()

    max_children_nodes = characterization_df[characterization_df["num_children"] == max_children]

    print(max_children_nodes[
        ["concept_id", "label", "num_children"]
    ])


    plt.figure(figsize=(8,5))
    plt.hist(characterization_df["min_depth"], bins=30)
    plt.hist(characterization_df["max_depth"], bins=30, alpha=0.7)
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title("Distribution of Concept Min-Max Depth")
    plt.legend(["Min Depth", "Max Depth"])
    plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(characterization_df["num_paths"], bins=100)
    plt.xlabel("Number of Paths")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Root-to-Concept Paths")
    plt.show()

    leaf_counts = characterization_df["is_leaf"].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(leaf_counts, labels=["Leaf", "Non-Leaf"], autopct="%1.1f%%")
    plt.title("Leaf vs Non-Leaf Concepts")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(characterization_df["max_depth"], characterization_df["num_paths"], alpha=0.5)
    plt.xlabel("Maximum Depth")
    plt.ylabel("Number of Paths")
    plt.title("Depth vs Polyhierarchy")
    plt.show()

    import numpy as np

    x = np.sort(characterization_df["num_paths"])
    y = 1.0 - np.arange(len(x)) / len(x)

    plt.figure(figsize=(8,5))

    plt.plot(x, y)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of Paths")
    plt.ylabel("P(X >= x)")

    plt.title("CCDF of Number of Paths")

    plt.show()