from extraction.extractor_factory import get_extractor

def run_extraction(text, extractor_name):
    extractor = get_extractor(extractor_name)
    return extractor.extract(text)

def run_multiple_extractors(text, extractor_names):
    results = {}

    for name in extractor_names:
        extractor = get_extractor(name)
        results[name] = extractor.extract(text)

    return results