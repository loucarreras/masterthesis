from transformers import pipeline
from .base_extractor import BaseExtractor

class TransformerExtractor(BaseExtractor):
    def __init__(self, model_name):
        super().__init__("transformer")
        self.ner = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple"
        )

    def extract(self, text):
        results = self.ner(text)
        return [
            {
                "text": r["word"],
                "start": r["start"],
                "end": r["end"],
                "label": r["entity_group"],
                "score": r["score"],
                "source": self.name
            }
            for r in results
        ]