import pandas as pd
from ontology.snomed_loader import SNOMEDHierarchy
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if os.path.exists("descriptions_ancestors_characterization_all.csv"):
        
        print("Loading characterization results...")
        characterization_df = pd.read_csv("descriptions_ancestors_characterization_all.csv", dtype={"concept_id": str})
        print(f"Loaded {len(characterization_df)} characterization results.")

    else:
        # LOAD SNOMED HIERARCHY
        print("Loading SNOMED CT hierarchy...")
        snomed = SNOMEDHierarchy(
            concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt",
            relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt",
            description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
        )
        print("SNOMED CT hierarchy loaded.")

        print("Loading SNOMED CT refset...")
        snomed.lang_refset_loader(r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Refset\Language\der2_cRefset_LanguageSnapshot-en_INT_20260501.txt")
        print(f"Loaded {len(snomed.lang_refset)} SNOMED CT refset entries.")

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

                descriptions = snomed.get_all_descriptions(ancestor)
                synonyms = snomed.get_all_synonyms(ancestor)
                preferred = snomed.get_preferred_terms(ancestor)
                acceptable = snomed.get_acceptable_terms(ancestor)

                rows.append(
                    {
                        "concept_id": ancestor,
                        "label": snomed.get_label(ancestor),
                        "descriptions": descriptions,
                        "num_descriptions": len(descriptions),
                        "synonyms": synonyms,
                        "num_synonyms": len(synonyms),
                        "preferred_terms": preferred,
                        "num_preferred_terms": len(preferred),
                        "acceptable_terms": acceptable,
                        "num_acceptable_terms": len(acceptable)
                    }
                )
        
        characterization_df = pd.DataFrame(rows)
        characterization_df.to_csv("descriptions_ancestors_characterization_all.csv", index=False, encoding="utf-8")
    
    print("Characterization summary:")
    print(characterization_df.head(10))
    print(characterization_df.describe(include="all").transpose())


    plt.figure(figsize=(8,5))
    plt.hist(characterization_df["num_descriptions"], bins=35)
    plt.xlabel("Number of Descriptions")
    plt.ylabel("Frequency")
    plt.title("Distribution of Descriptions per Concept")
    plt.show()