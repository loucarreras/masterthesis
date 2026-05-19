import medspacy
from .base_extractor import BaseExtractor

class MedSpaCyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__("medspacy")
        self.nlp = medspacy.load()

    def extract(self, text):
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "score": 1.0,
                "source": self.name
            }
            for ent in doc.ents
        ]