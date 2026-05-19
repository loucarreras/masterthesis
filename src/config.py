from pathlib import Path

# Paths
DATA_DIR = Path("../data/snomed-ct-entity-linking-challenge-1.2.1")
TRAIN_NOTES_PATH = DATA_DIR / "train_notes.csv"
TRAIN_ANN_PATH = DATA_DIR / "train_annotations.csv"

OUTPUT_DIR = Path("./outputs")

# Pipeline settings
MODEL_NAME = "en_core_sci_sm"
IOU_THRESHOLD = 0.05

# Debug
DEBUG = False
DEBUG_SIZE = 10

# Extraction
CONTEXT_WINDOW = 50
OFFSET_FIX = 0  # <-- fix this properly later