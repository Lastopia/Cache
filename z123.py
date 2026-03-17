from pathlib import Path
import pandas as pd
import spacy
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet, stopwords
import pyinflect

# ==================== 配置常量 ====================
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_VALUES = {lvl: i * 0.2 for i, lvl in enumerate(CEFR_LEVELS)}
BATCH_SIZE = 1024
SIMILARITY_THRESHOLD = 0.5
MIN_DISTANCE = 0.2

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    STOPWORDS = set(stopwords.words('english'))

class CEFRManager:
    def __init__(self, path="data.csv"):
        self.df = self.load_training_data(path)
        self.word_cefr_freq = {lvl: Counter() for lvl in CEFR_LEVELS}
        self.word_weighted_scores = {} 
        
        try:
            self.model = spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download("en_core_web_md")
            self.model = spacy.load("en_core_web_md")
            
        self.build_statistics()

    def load_training_data(self, path):
        file_path = Path(path)
        if not file_path.exists(): raise FileNotFoundError("data.csv not found.")
        return pd.read_csv(file_path)
    
    def softmax(self, x):
        """标准 Softmax 实现"""
        if np.all(x == 0): return np.zeros_like(x)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def build_statistics(self):
        """计算语料库中每个词的加权 CEFR 分数"""
        pipes = self.model.pipe(self.df['text'].astype(str), batch_size=BATCH_SIZE, disable=["ner"])
        for doc, cefr in zip(pipes, self.df['cefr_level']):
            tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            self.word_cefr_freq[cefr].update(tokens)

        all_words = set()
        for lvl in CEFR_LEVELS: all_words.update(self.word_cefr_freq[lvl].keys())

        for word in all_words:
            freqs = np.array([self.word_cefr_freq[lvl][word] for lvl in CEFR_LEVELS], dtype=float)
            total_freq = freqs.sum()
            
            if total_freq == 0:
                self.word_weighted_scores[word] = 0.0
            else:
                # --- 核心修改：先算百分比，再算 Softmax ---
                percentages = freqs / total_freq
                weights = self.softmax(percentages)
                # ---------------------------------------
                
                weighted_score = sum(weights[i] * CEFR_VALUES[CEFR_LEVELS[i]] for i in range(len(CEFR_LEVELS)))
                self.word_weighted_scores[word] = weighted_score

    def check(self, word):
        """交互式 Check 函数：反映改进后的百分比 Softmax 逻辑"""
        word = word.lower()
        freqs = np.array([self.word_cefr_freq[lvl][word] for lvl in CEFR_LEVELS], dtype=float)
        total_freq = freqs.sum()
        
        print(f"\n{'='*15} CEFR Analysis: '{word}' {'='*15}")
        if total_freq == 0:
            print(f"Result: Word '{word}' not found in data.csv")
            return

        percentages = freqs / total_freq
        weights = self.softmax(percentages)
        score = self.word_weighted_scores.get(word, 0.0)

        print(f"{'Level':<8} | {'Freq':<6} | {'Pct %':<8} | {'Softmax Weight':<15}")
        print("-" * 55)
        for i, lvl in enumerate(CEFR_LEVELS):
            print(f"{lvl:<8} | {int(freqs[i]):<6} | {percentages[i]*100:<7.2f}% | {weights[i]:<15.4f}")
        
        print("-" * 55)
        print(f"Final Weighted CEFR Score: {score:.4f}")
        closest_lvl = min(CEFR_LEVELS, key=lambda l: abs(CEFR_VALUES[l] - score))
        print(f"Mapped Category: {closest_lvl}")

    def get_best_synset_lesk(self, token, context_lemmas):
        lemma = token.lemma_.lower()
        wn_pos = {"VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV, "NOUN": wordnet.NOUN}.get(token.pos_)
        if not wn_pos: return None
        synsets = wordnet.synsets(lemma, pos=wn_pos)
        if not synsets: return None
        
        context_set = set(context_lemmas) - STOPWORDS
        best_ss, max_overlap = synsets[0], -1
        for ss in synsets:
            signature = set(ss.definition().lower().replace(',', '').split())
            for example in ss.examples():
                signature.update(example.lower().replace(',', '').split())
            overlap = len(signature.intersection(context_set))
            if overlap > max_overlap:
                max_overlap, best_ss = overlap, ss
        return best_ss

    def get_replacement(self, ori_token, context_lemmas, tar_cefr_val):
        old_lemma = ori_token.lemma_.lower()
        old_score = self.word_weighted_scores.get(old_lemma, 1.0)
        
        candidates = set()
        best_ss = self.get_best_synset_lesk(ori_token, context_lemmas)
        
        synsets = [best_ss] if best_ss else wordnet.synsets(old_lemma)
        for ss in synsets:
            for l in ss.lemmas():
                cand = l.name().replace('_', ' ').lower()
                if cand != old_lemma: candidates.add(cand)
            if hasattr(ss, 'hypernyms'):
                for hyper in ss.hypernyms():
                    for l in hyper.lemmas():
                        cand = l.name().replace('_', ' ').lower()
                        if cand != old_lemma: candidates.add(cand)

        if not candidates: return None

        cand_list = list(candidates)
        cand_docs = list(self.model.pipe(cand_list))
        valid_options = []

        print(f"\n[DEBUG] Word: '{ori_token.text}' (Score: {old_score:.3f}) -> Target: {tar_cefr_val:.2f}")
        print(f"{'Candidate':<18} | {'Simil.':<8} | {'Score':<8} | {'Dist.':<8} | {'Status'}")
        print("-" * 65)

        for cand_str, c_doc in zip(cand_list, cand_docs):
            if not ori_token.has_vector or not c_doc[0].has_vector: continue
            
            sim = ori_token.similarity(c_doc[0])
            cand_score = self.word_weighted_scores.get(cand_str)
            
            if cand_score is None:
                print(f"{cand_str:<18} | {sim:<8.2f} | {'N/A':<8} | {'N/A':<8} | SKIP: No Data")
                continue

            dist = abs(cand_score - tar_cefr_val)
            status = "OK"
            if sim < SIMILARITY_THRESHOLD: status = "SKIP: Sim"
            elif cand_score >= old_score: status = "SKIP: Harder"
            elif dist > MIN_DISTANCE: status = "SKIP: Dist"

            print(f"{cand_str:<18} | {sim:<8.2f} | {cand_score:<8.3f} | {dist:<8.3f} | {status}")

            if status == "OK":
                valid_options.append((cand_str, cand_score))

        if not valid_options:
            return None

        best_syn, best_score = min(valid_options, key=lambda x: abs(x[1] - tar_cefr_val))
        print(f"==> SELECTED: '{best_syn}' (Final Score: {best_score:.3f})")

        tag = ori_token.tag_
        original_lemma_hash = ori_token.lemma
        ori_token.lemma_ = best_syn
        replacement_text = ori_token._.inflect(tag)
        ori_token.lemma = original_lemma_hash
        
        return (replacement_text if replacement_text else best_syn) + ori_token.whitespace_

    def transform(self, sentence, target_level):
        doc = self.model(sentence)
        context_lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
        tar_cefr_val = CEFR_VALUES.get(target_level, 0.2)
        
        transformed = []
        for token in doc:
            if token.is_alpha and token.pos_ in ["VERB", "ADJ", "ADV", "NOUN"]:
                rep = self.get_replacement(token, context_lemmas, tar_cefr_val)
                transformed.append(rep if rep else token.text_with_ws)
            else:
                transformed.append(token.text_with_ws)
        return "".join(transformed).strip()

# ==================== 全局入口 ====================
manager = None

def transform_sentence(sentence, source_level, target_level):
    global manager
    if manager is None:
        manager = CEFRManager("data.csv")
    
    if source_level == target_level:
        result = sentence
    else:
        result = manager.transform(sentence, target_level)
    
    print(f"\nFinal Result: {result}")
    
    # 交互式 Check
    while True:
        user_word = input("\n[Check Mode] 输入单词查询分布 (或 'end' 退出): ").strip().lower()
        if user_word == 'end': break
        if user_word: manager.check(user_word)
            
    return result