from pathlib import Path
import pandas as pd
import spacy
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet
import pyinflect

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
BATCH_SIZE = 1024
MIN_APPEARANCE_COUNT = 10

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
        # 存储单词在各等级中的出现百分比
        self.word_cefr_percentage = defaultdict(lambda: {level: 0.0 for level in CEFR_LEVELS})
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
            total_count = sum(level_counts.values())
            if total_count > 0:
                for level in CEFR_LEVELS:
                    self.word_cefr_percentage[word][level] = level_counts.get(level, 0) / total_count

            best_level = max(CEFR_LEVELS, key=lambda x: (level_counts.get(x, 0), -CEFR_LEVELS.index(x)))
            if level_counts.get(best_level, 0) > 0:
                self.word_best_cefr[word] = best_level

    def check(self, token):
        word_str = token.lemma_.lower() if hasattr(token, 'lemma_') else str(token).lower()
        percentages = self.word_cefr_percentage.get(word_str)
        if not percentages:
            return
        print(f"Token: {word_str}")
        for level in CEFR_LEVELS:
            pct = percentages[level] * 100
            print(f"{level}-{pct:>6.2f}%  ", end="")
        print("")

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
            # 1. 获取直接同义词 (Lemmas)
            for i in syn.lemmas():
                candidate = i.name().replace('_', ' ').lower()
                if candidate != lemma:
                    synonyms.add(candidate)
            
            # 2. 获取上位词 (Hypernyms)
            for hyper in syn.hypernyms():
                for i in hyper.lemmas():
                    candidate = i.name().replace('_', ' ').lower()
                    if candidate != lemma:
                        synonyms.add(candidate)
                        
        # print("===================列举同义词中====================")
        # for syn in synonyms:
        #     self.check(syn)
        return synonyms

    def get_replacement(self, ori_token, tar_cefr, if_down):
        old_lemma = ori_token.lemma_.lower()
        pos = ori_token.pos_
        tag = ori_token.tag_
        
        synonyms = self.get_synonyms(old_lemma, pos)
        if not synonyms: return None
        
        candidates = list(synonyms) + [old_lemma]

        valid_candidates = []
        for cand in candidates:
            # 统计该候选词在 data.csv 中的全等级总出现次数
            total_cand_count = sum(self.cefr_word_counts[lvl].get(cand, 0) for lvl in CEFR_LEVELS)
            
            # 如果候选词够活跃，或者是原词本身，则视为有效候选
            if total_cand_count >= MIN_APPEARANCE_COUNT or cand == old_lemma:
                valid_candidates.append(cand)
        
        if not valid_candidates: return None
        
        # 从有效候选词中找到目标等级占比最高的
        best_syn = max(valid_candidates, key=lambda x: self.word_cefr_percentage[x].get(tar_cefr.name, 0))
        
        if best_syn == old_lemma:
            return None
        # --- 优化结束 ---

        # 变形逻辑
        original_lemma_hash = ori_token.lemma
        ori_token.lemma_ = best_syn
        replacement_text = ori_token._.inflect(tag)
        ori_token.lemma = original_lemma_hash
        
        if not replacement_text:
            replacement_text = best_syn
        
        return replacement_text + ori_token.whitespace_

    def transform(self, sentence, src_cefr, tar_cefr):
        doc = self.model(sentence)
        src_cefr = CEFRLevel(src_cefr)
        tar_cefr = CEFRLevel(tar_cefr)
        if_down = src_cefr > tar_cefr
        
        transformed_tokens = []
        for token in doc:
            if token.is_alpha:
                # self.check(token)
                rep = self.get_replacement(token, tar_cefr, if_down)
                transformed_tokens.append(rep if rep else token.text_with_ws)
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