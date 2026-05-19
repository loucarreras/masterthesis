import yake
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

text = notes_texts[0]
print(text)

# Simple usage with default parameters
kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(text)

for kw, score in keywords:
    print(f"{kw} ({score})")

# With custom parameters
custom_kw_extractor = yake.KeywordExtractor(
    lan="en",              # language
    n=3,                   # ngram size
    dedupLim=0.9,          # deduplication threshold
    dedupFunc='seqm',      # deduplication function
    windowsSize=1,         # context window
    top=10,                # number of keywords to extract
    features=None          # custom features
)

keywords = custom_kw_extractor.extract_keywords(text)

# src/extraction/yake_extractor.py

import yake
from .base_extractor import BaseExtractor

class YAKEExtractor(BaseExtractor):
    
    def __init__(self, top_k=10):
        self.kw_extractor = yake.KeywordExtractor(top=top_k)
    
    def extract(self, text: str):
        keywords = self.kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
        # kw_extractor = yake.KeywordExtractor()
        # keywords_list = kw_extractor.extract_keywords(text)
        # keywords = [kw[0] for kw in keywords_list]