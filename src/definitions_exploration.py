import pandas as pd
from ontology.snomed_loader import SNOMEDHierarchy
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
if __name__ == "__main__":

    # LOAD SNOMED HIERARCHY
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt",
        relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt",
        description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
    )
    print("SNOMED CT hierarchy loaded.")

    print("Loading SNOMED CT definitions...")
    snomed.load_text_definitions(r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_TextDefinition_Snapshot-en_INT_20260501.txt")
    print(f"Loaded {len(snomed.text_definitions)} SNOMED CT definitions.")

    if os.path.exists("definitions_ancestors_characterization.csv"):
        
        print("Loading characterization results...")
        characterization_df = pd.read_csv("definitions_ancestors_characterization.csv", dtype={"concept_id": str})
        print(f"Loaded {len(characterization_df)} characterization results.")

    else:
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

                has_def = snomed.has_text_definition(ancestor)
                definition = snomed.get_text_definitions(ancestor) if has_def else None
                num_defs = snomed.get_num_text_definitions(ancestor) if has_def else None
                def_length = snomed.get_text_definition_length(ancestor) if has_def else None
                
                rows.append(
                    {
                        "concept_id": ancestor,
                        "label": snomed.get_label(ancestor),
                        "is_leaf": snomed.is_leaf(ancestor),
                        "has_text_definition": has_def,
                        "text_definitions": definition,
                        "num_text_definitions": num_defs,
                        "text_definition_length": def_length
                    }
                )

        characterization_df = pd.DataFrame(rows)
        characterization_df.to_csv("definitions_ancestors_characterization.csv", index=False, encoding="utf-8")
    
    print("Characterization summary:")
    print(characterization_df.head(10))
    print(characterization_df.describe(include="all").transpose())


    if os.path.exists("snomed_text_definitions_characterization.csv"):
        print("Loading characterization results...")
        definitions_df = pd.read_csv("snomed_text_definitions_characterization.csv", dtype={"concept_id": str})
    
    else:
        rows = []
        for concept_id in snomed.text_definitions["conceptId"]:
            rows.append(
                {
                    "concept_id": concept_id,
                    "label": snomed.get_label(concept_id),
                    "text_definition": snomed.get_text_definitions(concept_id),
                    "num_text_definitions": snomed.get_num_text_definitions(concept_id),
                    "text_definition_length": snomed.get_text_definition_length(concept_id),
                    "is_leaf": snomed.is_leaf(concept_id),
                    "num_parents": len(snomed.get_parents(concept_id)),
                    "num_children": len(snomed.get_children(concept_id)),
                }
            )
        
        definitions_df = pd.DataFrame(rows)
        definitions_df.to_csv("snomed_text_definitions_characterization.csv", index=False, encoding="utf-8")

    print("Characterization of text definitions in SNOMED CT:")    

    print("Number of available text definitions:", len(snomed.text_definitions))

    print(snomed.text_definitions.describe(include="all").transpose())

    print(definitions_df.describe(include="all").transpose())

    plt.figure(figsize=(8,5))
    plt.hist(characterization_df["num_text_definitions"], bins=3)
    plt.xlabel("Number of Text Definitions")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Text Definitions")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.boxplot(definitions_df["num_children"])
    plt.xlabel("Concepts with Text Definitions")
    plt.ylabel("Number of Children")
    plt.title("Distribution of Child Nodes")
    plt.show()
    
    with_def_leafs = characterization_df[(characterization_df["has_text_definition"] == True) & (characterization_df["is_leaf"] == True)]
    with_def_no_leafs  = characterization_df[(characterization_df["has_text_definition"] == True) & (characterization_df["is_leaf"] == False)]
    no_def_leafs = characterization_df[(characterization_df["has_text_definition"] == False) & (characterization_df["is_leaf"] == True)]
    no_def_no_leafs = characterization_df[(characterization_df["has_text_definition"] == False) & (characterization_df["is_leaf"] == False)]

    matrix_def_leafs = np.array([with_def_leafs.shape[0], with_def_no_leafs.shape[0], no_def_leafs.shape[0], no_def_no_leafs.shape[0]]).reshape(2,2)
    
    heatmap_data = characterization_df.groupby(["has_text_definition", "is_leaf"]).size().unstack(fill_value=0)

    plt.figure(figsize=(8,5))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Is Leaf")
    plt.ylabel("Has Text Definition")
    plt.title("Distribution of Concepts")
    plt.show()