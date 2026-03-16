from pathlib import Path
import pandas as pd
import spacy
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet
import pyinflect

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
BATCH_SIZE = 1024

# Convenient for direct comparison of CEFR levels
class CEFRLevel:
    def __init__(self, cefr):
        if cefr not in CEFR_LEVELS:
            # default cefr for unkown
            self.name = "A1"
        else:
            self.name = cefr
        self.index = CEFR_LEVELS.index(self.name)

    def __gt__(self, other): return self.index > other.index
    def __lt__(self, other): return self.index < other.index
    def __ge__(self, other): return self.index >= other.index
    def __le__(self, other): return self.index <= other.index
    def __eq__(self, other): return self.index == other.index
    
    def __str__(self): return self.name

class CEFRManager:
    def __init__(self, path="data.csv"):
        self.df = self.load_training_data(path)
        self.cefr_word_counts = {level: Counter() for level in CEFR_LEVELS}
        self.word_best_cefr = defaultdict(lambda: "A1") 
        self.model = spacy.load("en_core_web_sm")
        self.create_tables()

    def load_training_data(self, path):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError("data.csv not found.")
        df = pd.read_csv(file_path)
        return df

    def create_tables(self):
        pipes = self.model.pipe(self.df['text'].astype(str), batch_size=BATCH_SIZE, disable=["ner"])
        word_cefr_freq = defaultdict(lambda: defaultdict(int))
        
        for doc, cefr in zip(pipes, self.df['cefr_level']):
            tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            self.cefr_word_counts[cefr].update(tokens)
            for t in tokens:
                word_cefr_freq[t][cefr] += 1

        for word, level_counts in word_cefr_freq.items():
            best_level = max(CEFR_LEVELS, key=lambda x: (level_counts.get(x, 0), -CEFR_LEVELS.index(x)))
            if level_counts.get(best_level, 0) > 0:
                self.word_best_cefr[word] = best_level

    def get_cefr(self, word_str):
        lvl_str = self.word_best_cefr.get(word_str.lower(), "A1")
        return CEFRLevel(lvl_str)

    def get_synonyms(self, lemma, pos):
        match pos:
            case "NOUN": pos = wordnet.NOUN
            case "VERB": pos = wordnet.VERB
            case "ADJ":  pos = wordnet.ADJ
            case "ADV":  pos = wordnet.ADV
            case _: return set()

        synonyms = set()
        for syn in wordnet.synsets(lemma, pos=pos):
            for i in syn.lemmas():
                candidate = i.name().replace('_', ' ').lower()
                if candidate != lemma:
                    synonyms.add(candidate)
        return synonyms

    def get_replacement(self, ori_token, tar_cefr, if_down):
        lemma = ori_token.lemma_.lower()
        pos = ori_token.pos_
        tag = ori_token.tag_
        
        synonyms = self.get_synonyms(lemma, pos)
        if not synonyms: return None
        
        candidates = []
        for syn in synonyms:
            syn_cefr = self.get_cefr(syn)
            if if_down and syn_cefr <= tar_cefr:
                candidates.append(syn)
            elif not if_down and syn_cefr >= tar_cefr:
                candidates.append(syn)
                
        if not candidates: return None
        
        best_syn = max(candidates, key=lambda x: self.cefr_word_counts[tar_cefr.name].get(x, 0))
        replacement = ori_token._.inflect(tag, form_num=0) or best_syn
        
        return replacement + ori_token.whitespace_

    def transform(self, sentence, src_cefr, tar_cefr):
        doc = self.model(sentence)
        src_cefr = CEFRLevel(src_cefr)
        tar_cefr = CEFRLevel(tar_cefr)
        if_down = src_cefr > tar_cefr
        
        transformed_tokens = []
        for token in doc:
            if token.is_alpha:
                cur_cefr = self.get_cefr(token.lemma_)

                needs_change = (if_down and cur_cefr > tar_cefr) or (not if_down and cur_cefr < tar_cefr)
                
                if needs_change:
                    rep = self.get_replacement(token, tar_cefr, if_down)
                    transformed_tokens.append(rep if rep else token.text_with_ws)
                else:
                    transformed_tokens.append(token.text_with_ws)
            else:
                transformed_tokens.append(token.text_with_ws)

        return "".join(transformed_tokens).strip()


manager = None

def transform_sentence(sentence, source_level, target_level):
    global manager
    if manager is None:
        manager = CEFRManager("data.csv")
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    

    if source_level == target_level:
        return sentence        
    return manager.transform(sentence, source_level, target_level)
