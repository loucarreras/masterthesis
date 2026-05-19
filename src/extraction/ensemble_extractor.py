from extraction.base_extractor import BaseExtractor

class EnsembleExtractor(BaseExtractor):
    def __init__(self, extractors, merge_fn):
        super().__init__("ensemble")
        self.extractors = extractors
        self.merge_fn = merge_fn

    def extract(self, text):
        all_entities = []
        for ext in self.extractors:
            all_entities.append(ext.extract(text))

        return self.merge_fn(all_entities)