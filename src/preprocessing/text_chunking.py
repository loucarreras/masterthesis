import pandas as pd
import re
import scispacy
import spacy
from typing import List, Dict
import unicodedata
# Load SciSpaCy model
nlp = spacy.load("en_core_sci_scibert")

# Clean text
def clean_text(text: str) -> str:
    """
    Clean text by removing de-identification brackets, normalizing whitespace, and removing non-ascii characters
    """
    
    if pd.isna(text):
        return ""
    
    # Remove de-identification brackets
    text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
    # text = re.sub(r'_+', ' ', text)

    # Normalize whitespace
    text = re.sub(r"\r", "\n", text)
    #text = re.sub(r"\n+", "\n", text)
    #text = re.sub(r"[ \t]+", " ", text)

    text = unicodedata.normalize("NFKC", text)
    
    return text.strip()

# Section splitting
def split_by_sections(text: str) -> List[Dict]:
    """
    Returns list of {section_title, content, start, end}
    """
    pattern = r"\n\s*([A-Za-z][A-Za-z\s]{2,}:)\s*\n" # Finding section titles
    parts = re.split(pattern, text)

    sections = []

    # No section fallback
    if len(parts) < 3:
        return [{
            "section": "FULL_TEXT",
            "text": text,
            "start": 0,
            "end": len(text)
        }]
    
    cursor = 0

    for i in range(1, len(parts), 2):
        title = parts[i].strip(": \n")
        content = parts[i + 1]

        #find section start in the original text
        start_idx = text.find(content, cursor)
        if start_idx == -1:
            continue

        end_idx = start_idx + len(content)
        cursor = end_idx

        if content.strip():
            sections.append({
                "section": title,
                "text": content.strip(),
                "start": start_idx,
                "end": end_idx
            })

    return sections

# Alternatively, split into chunks of sentences with overlap
def split_into_sentence_chunks(text: str, max_words=180, overlap_words=40):
    """Splits text into chunks of sentences with a specified maximum number of words and overlap."""
    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        sentences.append({
            "text": sent.text.strip(),
            "start": sent.start_char,
            "end": sent.end_char
        })

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent["text"].split()
        sent_len = len(words)

        if current_len + sent_len <= max_words:
            current.append(sent)
            current_len += sent_len
        else:
            # finalize chunk
            chunk_text = " ".join([s["text"] for s in current])

            chunks.append({
                "text": chunk_text,
                "start": current[0]["start"],
                "end": current[-1]["end"]
            })

            # overlap logic
            overlap = current[-overlap_words:] if len(current) > overlap_words else current

            current = overlap + [sent]
            current_len = sum(len(s["text"].split()) for s in current)

    if current:
        chunks.append({
            "text": " ".join([s["text"] for s in current]),
            "start": current[0]["start"],
            "end": current[-1]["end"]
        })

    return chunks

# Hybrid chunking strategy
def chunk_text(text: str,  max_section_length: int = 1000) -> List[Dict]:
    """
    Chunk text by sections, and further split long sections into sentence chunks.
    """
    sections = split_by_sections(text)
    final_chunks = []

    for sec in sections:
        section_name = sec["section"]
        content = sec["text"]
        sec_start = sec["start"]
        sec_end = sec["end"]

        # If section is short, keep as is
        if len(content.split()) <= max_section_length:
            if len(content.split()) >= 3:  # Filter out very short sections
                final_chunks.append({
                    "section": section_name,
                    "chunk": content,
                    "start": sec_start,
                    "end": sec_end,
                    "chunk_start": sec_start,
                    "chunk_end": sec_end
                })
        else:
            # Split into sentence chunks
            sub_chunks = split_into_sentence_chunks(content)
            for sub in sub_chunks:
                if len(sub["text"].split()) >= 3: # Filter out very short sections
                    final_chunks.append({
                        "section": section_name,
                        "chunk": sub["text"],
                        "start": sec_start,
                        "end": sec_end,
                        "chunk_start": sec_start + sub["start"],
                        "chunk_end": sec_start + sub["end"]
                    })

    return final_chunks

def preprocess_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: MIMIC dataframe with at least columns ["note_id", "text"]
    Output: chunked dataframe
    """

    records = []

    for _, row in df.iterrows():
        note_id = row.get("note_id", None)
        raw_text = row.get("text", "")

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)

        for i, ch in enumerate(chunks):
            records.append({
                "note_id": note_id,
                "section": ch["section"],
                "chunk_id": i,
                "text": ch["chunk"],
                "section_start": ch["start"],
                "section_end": ch["end"],
                "chunk_start": ch["chunk_start"],
                "chunk_end": ch["chunk_end"]
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading data...")

    paths = [
        r"C:\DESTINATION\note\discharge.csv.gz",
        #r"C:\DESTINATION\note\radiology.csv.gz",
    ]

    dfs = [pd.read_csv(p, compression='gzip') for p in paths]
    notes_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(notes_df)} notes.")

    selected_note_id = "10043750-DS-6"
    selected_note = notes_df[notes_df["note_id"] == selected_note_id]

    processed_df = preprocess_notes(selected_note)

    print(f"Dimensions of processed dataframe: {processed_df.shape}")
    print(processed_df)
