import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SNOMEDIndex:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.terms_df = None

    def build(self, terms_df):

        self.terms_df = terms_df

        embeddings = self.model.encode(
            terms_df["term"].tolist(),
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=5):

        q = self.model.encode([query]).astype("float32")

        D, I = self.index.search(q, top_k)

        rows = self.terms_df.iloc[I[0]]
        return rows.to_dict("records")