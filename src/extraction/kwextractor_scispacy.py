# src/extraction/kwextractor_scispacy.py

import spacy
from .base_extractor import BaseExtractor

class SciSpaCyExtractor(BaseExtractor):

    def __init__(self, model="en_core_sci_sm"):
        self.nlp = spacy.load(model)

    def extract(self, text: str):
        doc = self.nlp(text)

        return [
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
            for ent in doc.ents
        ]