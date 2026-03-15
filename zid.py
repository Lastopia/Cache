from pathlib import Path
import pandas as pd
import spacy
from collections import Counter, defaultdict


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")
    df = pd.read_csv(file_path)
    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df

# -------------------------------------------------
# 1. Builld the Vocabulary and Frequency Tables
# -------------------------------------------------
BATCH_SIZE = 1024

class Assignment:
    def __init__(self,path = "data.csv"):
        self.df = load_training_data(path)
        self.cefr_word_counts = {level: Counter() for level in CEFR_LEVELS}
        self.word2cefr = defaultdict(lambda: defaultdict(int))
        self.model = spacy.load("en_core_web_sm")
        # self.model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    def create_tables(self):
        for text, cefr in zip(self.model.pipe(self.df['text'].astype(str), batch_size=BATCH_SIZE), self.df['cefr_level']):
            tokens = [token.lemma_.lower() for token in text if token.is_alpha]
            self.cefr_word_counts[cefr].update(tokens)

        for cefr, counter in self.cefr_word_counts.items():
            for word, freq in counter.items():
                self.word2cefr[word][cefr] += freq

        for word, level_counts in self.word2cefr.items():
            best_level = max(
                CEFR_LEVELS,
                key=lambda x: (level_counts.get(x, 0), -CEFR_LEVELS.index(x))
            )
            if level_counts.get(best_level, 0) > 0:
                self.word2cefr[word] = best_level









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
