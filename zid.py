# Rename this file to your zID, e.g. z1234567.py

from pathlib import Path
import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


# -------------------------------------------------
# Optional: Load training data
# -------------------------------------------------


def load_training_data(path="data.csv"):
    """
    Loads the provided training dataset.

    Expected columns:
    - text
    - cefr_level
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)

    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")

    return df


# -------------------------------------------------
# Optional: Initialise resources globally
# -------------------------------------------------

# Students may:
# - Load and preprocess data here
# - Train a lightweight model
# - Build lookup tables
# - Create rule-based mappings


# -------------------------------------------------
# Required Function
# -------------------------------------------------


def transform_sentence(sentence, source_level, target_level):
    """
    Transform a sentence from source CEFR level to target CEFR level.

    Parameters:
        sentence (str): Input sentence.
        source_level (str): CEFR level of the input sentence.
        target_level (str): Target CEFR level.

    Returns:
        str: Transformed sentence.
    """

    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")

    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    # ---------------------------------------------
    # IMPLEMENT YOUR SOLUTION BELOW
    # ---------------------------------------------

    # Placeholder behaviour
    # Replace this with your actual transformation logic.
    return sentence
