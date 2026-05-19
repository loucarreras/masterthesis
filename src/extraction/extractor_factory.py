from extraction.kwextractor_scispacy import SciSpaCyExtractor
from extraction.kwextractor_medspacy import MedSpaCyExtractor
from extraction.kwextractor_transformers import TransformerExtractor
from extraction.kwextractor_yake import YAKEExtractor

def get_extractor(name: str):
    if name == "scispacy":
        return SciSpaCyExtractor(model="en_core_sci_sm")

    elif name == "medspacy":
        return MedSpaCyExtractor()

    elif name == "transformer":
        return TransformerExtractor("d4data/biomedical-ner-all")

    elif name == "yake":
        return YAKEExtractor()

    else:
        raise ValueError(f"Unknown extractor: {name}")