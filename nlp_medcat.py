from medcat.cat import CAT
import os
import pandas as pd

paths = [
    r"C:\DESTINATION\note\discharge.csv.gz",
    #r"C:\DESTINATION\note\radiology.csv.gz",
]

dfs = [pd.read_csv(p, compression='gzip') for p in paths]
notes_df = pd.concat(dfs, ignore_index=True)

if 'text' in notes_df.columns:
    notes_texts = notes_df['text'].astype(str).tolist()
else:
    # fallback: concatenate all columns
    notes_texts = notes_df.astype(str).agg(' '.join, axis=1).tolist()

print(f"Loaded {len(notes_texts)} notes.")

clinical_note = notes_texts[0:2]

cat = CAT.load_model_pack("model_pack_path")

doc = cat(clinical_note[0:2])

for ent in doc.ents:
    print(ent.text, ent.cui, ent._.negex)