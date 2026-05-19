# utils.py

def merge_adjacent_entities(entities, text, max_gap=1):
    """
    Merge adjacent entities if the gap between them is <= max_gap.
    
    Args:
        entities (list of dict): Each dict must have 'term', 'start', 'end'
        text (str): The text from which the entities were extracted
        max_gap (int): Maximum allowed gap between entities to merge

    Returns:
        List of merged entities
    """
    if not entities:
        return []

    # Sort by start position
    entities = sorted(entities, key=lambda x: x["start"])
    
    merged = []
    current = entities[0].copy()

    for next_ent in entities[1:]:
        gap = next_ent["start"] - current["end"]
        between_text = text[current["end"]:next_ent["start"]]

        if gap <= max_gap and between_text.strip() == "":
            # Merge
            current["term"] = current["term"] + " " + next_ent["term"]
            current["end"] = next_ent["end"]
        else:
            merged.append(current)
            current = next_ent.copy()

    merged.append(current)
    return merged