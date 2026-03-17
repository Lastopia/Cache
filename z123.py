from pathlib import Path
import pandas as pd
import spacy
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet
import pyinflect

# ==================== 配置常量 ====================
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
# 非线性分值映射：拉开高低等级间距
CEFR_VALUES = np.array([0.0, 0.12, 0.35, 0.65, 0.88, 1.0])
CEFR_VAL_MAP = {lvl: CEFR_VALUES[i] for i, lvl in enumerate(CEFR_LEVELS)}

BATCH_SIZE = 1024
SIMILARITY_THRESHOLD = 0.5 
USE_RELATIVE_STEP = True 
POWER_EXPONENT = 2  # 幂运算突出赢家

class CEFRManager:
    def __init__(self, path="data.csv"):
        self.df = self.load_training_data(path)
        self.word_cefr_freq = {lvl: Counter() for lvl in CEFR_LEVELS}
        self.word_weighted_scores = {} 
        
        try:
            # 语义转换必须使用 md 或 lg 模型
            self.model = spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download("en_core_web_md")
            self.model = spacy.load("en_core_web_md")
            
        self.build_statistics()

    def load_training_data(self, path):
        file_path = Path(path)
        if not file_path.exists(): raise FileNotFoundError("data.csv not found.")
        return pd.read_csv(file_path)
    
    def power_normalize(self, pcts):
        """幂函数归一化"""
        p_pcts = np.power(pcts, POWER_EXPONENT)
        sum_p = p_pcts.sum()
        if sum_p == 0: return np.zeros_like(pcts)
        return p_pcts / sum_p

    def build_statistics(self):
        """核心统计逻辑：集成高阶熵惩罚"""
        pipes = self.model.pipe(self.df['text'].astype(str), batch_size=BATCH_SIZE, disable=["ner"])
        for doc, cefr in zip(pipes, self.df['cefr_level']):
            tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
            self.word_cefr_freq[cefr].update(tokens)

        all_words = set()
        for lvl in CEFR_LEVELS: all_words.update(self.word_cefr_freq[lvl].keys())

        # 最大熵常量 ln(6)
        H_MAX = np.log(len(CEFR_LEVELS))

        for word in all_words:
            freqs = np.array([self.word_cefr_freq[lvl][word] for lvl in CEFR_LEVELS], dtype=float)
            total = freqs.sum()
            if total == 0:
                self.word_weighted_scores[word] = 0.0
            else:
                pcts = freqs / total
                
                # 1. 计算香农熵
                p_nonzero = pcts[pcts > 0]
                entropy = -np.sum(p_nonzero * np.log(p_nonzero))
                
                # 2. 归一化熵因子 (0-1)
                entropy_factor = entropy / H_MAX
                
                # 3. 高阶惩罚逻辑 (4次方)
                # 当 entropy_factor 较小时（如 0.387），(0.387)^4 极小，Penalty 接近 1
                # 保护了局部波动的单词不被降级
                penalty = 1.0 - np.power(entropy_factor, 4)
                
                weights = self.power_normalize(pcts)
                raw_score = sum(weights[i] * CEFR_VALUES[i] for i in range(len(CEFR_LEVELS)))
                
                self.word_weighted_scores[word] = raw_score * penalty

    def check(self, word):
        """调试用：查看单词的熵分布与惩罚细节"""
        word = word.lower()
        freqs = np.array([self.word_cefr_freq[lvl][word] for lvl in CEFR_LEVELS], dtype=float)
        total = freqs.sum()
        print(f"\n{'='*15} CEFR Analysis (High-Order Entropy): '{word}' {'='*15}")
        if total == 0:
            print(f"Result: Word '{word}' not found.")
            return
        
        pcts = freqs / total
        weights = self.power_normalize(pcts)
        
        p_nz = pcts[pcts > 0]
        ent = -np.sum(p_nz * np.log(p_nz))
        ent_f = ent / np.log(6)
        pen = 1.0 - np.power(ent_f, 4) # 同步显示 4 次幂效果
        
        score = self.word_weighted_scores.get(word, 0.0)
        
        print(f"{'Level':<8} | {'Freq':<6} | {'Pct %':<8} | {'Power Weight'}")
        print("-" * 55)
        for i, lvl in enumerate(CEFR_LEVELS):
            print(f"{lvl:<8} | {int(freqs[i]):<6} | {pcts[i]*100:<7.2f}% | {weights[i]:<15.4f}")
        print("-" * 55)
        print(f"Normalized Entropy (0-1): {ent_f:.4f}")
        print(f"Penalty Factor (Power 4): {pen:.4f}")
        print(f"Final Weighted CEFR Score: {score:.4f}")
        closest = min(CEFR_LEVELS, key=lambda l: abs(CEFR_VAL_MAP[l] - score))
        print(f"Mapped Category: {closest}")

    def get_replacement(self, ori_token, target_cefr_val, target_step):
        """寻找语义相似且符合 CEFR 目标的替换词"""
        old_lemma = ori_token.lemma_.lower()
        old_score = self.word_weighted_scores.get(old_lemma, 1.0)
        
        candidates = set()
        wn_pos = {"VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV, "NOUN": wordnet.NOUN}.get(ori_token.pos_)
        if wn_pos:
            for ss in wordnet.synsets(old_lemma, pos=wn_pos):
                for l in ss.lemmas():
                    cand = l.name().replace('_', ' ').lower()
                    if cand != old_lemma: candidates.add(cand)
                for h in ss.hypernyms():
                    for l in h.lemmas():
                        cand = l.name().replace('_', ' ').lower()
                        if cand != old_lemma: candidates.add(cand)

        if not candidates: return None

        cand_list = list(candidates)
        cand_docs = list(self.model.pipe(cand_list))
        
        valid_options = []
        print(f"\n[DEBUG] Word: '{ori_token.text}' (Score: {old_score:.3f})")
        print(f"{'Candidate':<18} | {'Simil.':<8} | {'Score':<8} | {'DistToTar':<10} | {'Status'}")
        print("-" * 65)

        for cand_str, c_doc in zip(cand_list, cand_docs):
            cand_score = self.word_weighted_scores.get(cand_str)
            if cand_score is None: continue
            
            # 相似度计算
            sim = 0.0
            if ori_token.has_vector and c_doc[0].has_vector:
                sim = ori_token.similarity(c_doc[0])
            
            cand_step = cand_score - old_score
            is_valid_dir = (target_step * cand_step > 0) or (abs(target_step) < 0.05)
            dist_to_target = abs(cand_score - target_cefr_val)

            status = "OK"
            if sim < SIMILARITY_THRESHOLD: status = "SKIP:Sim"
            elif not is_valid_dir: status = "SKIP:Dir"

            print(f"{cand_str:<18} | {sim:<8.2f} | {cand_score:<8.3f} | {dist_to_target:<10.3f} | {status}")

            if status == "OK":
                valid_options.append((cand_str, dist_to_target))

        if not valid_options: return None

        best_syn, _ = min(valid_options, key=lambda x: x[1])
        print(f"==> SELECTED: '{best_syn}'")

        tag = ori_token.tag_
        original_lemma_hash = ori_token.lemma
        ori_token.lemma_ = best_syn
        replacement_text = ori_token._.inflect(tag)
        ori_token.lemma = original_lemma_hash
        return (replacement_text if replacement_text else best_syn) + ori_token.whitespace_

    def transform(self, sentence, source_level, target_level):
        doc = self.model(sentence)
        src_val = CEFR_VAL_MAP.get(source_level, 0.8)
        tar_val = CEFR_VAL_MAP.get(target_level, 0.2)
        target_step = tar_val - src_val
        
        print(f"\n--- Strategy: {source_level}({src_val}) -> {target_level}({tar_val}) ---")
        
        transformed = []
        for token in doc:
            if token.is_alpha and token.pos_ in ["VERB", "ADJ", "ADV", "NOUN"]:
                rep = self.get_replacement(token, tar_val, target_step)
                transformed.append(rep if rep else token.text_with_ws)
            else:
                transformed.append(token.text_with_ws)
        return "".join(transformed).strip()

# ==================== 入口 ====================
manager = None

def transform_sentence(sentence, source_level, target_level):
    global manager
    if manager is None: manager = CEFRManager("data.csv")
    result = manager.transform(sentence, source_level, target_level)
    print(f"\nFinal Result: {result}")
    while True:
        user_word = input("\n[Check Mode] Word (or 'end'): ").strip().lower()
        if user_word == 'end': break
        if user_word: manager.check(user_word)
    return result