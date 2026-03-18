from pathlib import Path
import pandas as pd
import spacy
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet
from pyinflect import getInflection

# ==================== Parameters ====================
DEBUG_MODE = True
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_VALUES = {"A1":0.0, "A2":0.2, "B1":0.4, "B2":0.6, "C1":0.8, "C2":1.0}
TAR_POS = {"VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV, "NOUN": wordnet.NOUN}

BATCH_SIZE = 1024
SIMILARITY_THRESHOLD = 0.5 

class CEFRManager:
    def __init__(self, path="data.csv"):
        try:
            self.model = spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download("en_core_web_md")
            self.model = spacy.load("en_core_web_md")
        self.df = self.load_training_data(path)
        # Structure is {"word":{"cefr":freq}}
        self.word_cefr_freq = self.get_freq_table()
        # Structure is {"word":score}
        self.cefr_scores = self.get_cefr_scores()


    def load_training_data(self, path):
        path = Path(path)
        if not path.exists(): 
            raise FileNotFoundError("data.csv not found.")
        return pd.read_csv(path)
    
    def get_freq_table(self):
        # Create multiple pipes to conduct multiple thread
        freq_table = {}
        pipes = self.model.pipe(self.df["text"].astype(str), batch_size=BATCH_SIZE)

        for text, cefr in zip(pipes, self.df["cefr_level"]):
            tokens = [token.lemma_.lower() for token in text if token.is_alpha]
            for token in tokens:
                if token not in freq_table:
                    freq_table[token] = {cefr: 0 for cefr in CEFR_LEVELS}
                freq_table[token][cefr] += 1
        return freq_table

    def get_cefr_scores(self):
        # Create vocabulary list to record all words in the training data.
        # Tokens above is a local variable in each pipe. So we need a new global one.
        cefr_scores = {}
        vocab = self.word_cefr_freq.keys()
        for word in vocab:
            # Make it array to calculate conveniently
            freqs = np.array([self.word_cefr_freq[word][cefr] for cefr in CEFR_LEVELS], dtype=float)
            total = freqs.sum()
            distributions = freqs / total

            weights = self.power_normalize(distributions)
            penalty = self.entropy_penalty(distributions)
            raw_score = 0
            for i in range(len(weights)):
                raw_score += weights[i] * CEFR_VALUES[CEFR_LEVELS[i]]
            cefr_scores[word] = raw_score*penalty
        return cefr_scores
    
    def power_normalize(self, distributions):
        '''
        Use power normalize to highlight the may features.
        '''
        distributions = np.power(distributions, 2)
        sum = distributions.sum()
        return distributions / sum

    def entropy_penalty(self,distributions):
        '''
        Common words like 'buy' appear in all levels which lead its cefr score higher.
        Create entropy penaly coefficient to prevent that. 
        '''
        a = distributions[distributions > 0]
        entropy = -np.sum(a * np.log(a))
        penalty = 1.0 - np.power(entropy / np.log(len(CEFR_LEVELS)), 4) / 4
        return penalty

    def get_syn(self,lemma,tar_pos):
        '''
        Only find lemma of syn with the same pos
        '''
        candidates = set()
        for syn in wordnet.synsets(lemma, pos=tar_pos):
            for lem in syn.lemmas():
                cand = lem.name().replace('_', ' ').lower()
                if cand != lemma: candidates.add(cand)

            for hyper in syn.hypernyms():
                for lem in hyper.lemmas():
                    cand = lem.name().replace('_', ' ').lower()
                    if cand != lemma: candidates.add(cand)

        if not candidates: return [],[]
        syn_list = list(candidates)
        syn_docs = list(self.model.pipe(syn_list))
        return syn_list, syn_docs

    def get_replacement(self, token, target_cefr_val, tar_distance):
        '''
        Three considerations:
            1. Similarity > 0.5
            2. CEFR score change direction should be same as the requirement.
            3. The Distance between replacement and origin token should be minimal.
        '''
        lemma = token.lemma_.lower()        
        tar_pos = TAR_POS.get(token.pos_)
        cefr_score = self.cefr_scores.get(lemma, 1.0)
        valid_options = []

        syn_list,syn_docs = self.get_syn(lemma,tar_pos)
        if not syn_list: return None

        if DEBUG_MODE:
            print(f"\nWord: '{token.text}' (Score: {cefr_score:.3f})")
            print(f"{'Candidate':<18} | {'Simil.':<8} | {'Score':<8} | {'Distance':<10} | {'DistToTar':<10} | {'Status'}")
            print("-" * 65)

        for syn, doc in zip(syn_list, syn_docs):
            syn_score = self.cefr_scores.get(syn)
            # If no record in vocabulary,skip
            if syn_score is None: continue
            sim = 0.0
            if token.has_vector and doc[0].has_vector:
                sim = token.similarity(doc[0])
            
            distance = syn_score - cefr_score
            is_valid_dir = (tar_distance * distance > 0)
            dist_to_tar = abs(syn_score - target_cefr_val)
            status = "OK"
            if sim < SIMILARITY_THRESHOLD: status = "SKIP:Sim"
            elif not is_valid_dir: status = "SKIP:Dir"
            if status == "OK":
                valid_options.append((syn, dist_to_tar))

            if DEBUG_MODE:
                print(f"{syn:<18} | {sim:<8.2f} | {syn_score:<8.3f} |{distance:<10.3f} | {dist_to_tar:<10.3f} | {status}")



        if not valid_options: return None
        # Get the syn with min distance to target cefr score
        best_syn, _ = min(valid_options, key=lambda x: x[1])

        # Allow the replaced word to inherit the form of the previous word.
        tag = token.tag_
        res = getInflection(best_syn, tag)
        replacement = res[0] if res else best_syn
        # Ensure the token has the same number of whitespace following
        return replacement + token.whitespace_

    def transform(self, sentence, source_level, target_level):
        '''
        If have replacement, change it. 
        Or, use the original token.
        '''
        text = self.model(sentence)
        src_val = CEFR_VALUES.get(source_level)
        tar_val = CEFR_VALUES.get(target_level)
        tar_distance = tar_val - src_val        
        transformed = []
        for token in text:
            if token.is_alpha and token.pos_ in ["VERB", "ADJ", "ADV", "NOUN"]:
                rep = self.get_replacement(token, tar_val, tar_distance)
                transformed.append(rep if rep else token.text_with_ws)
            else:
                transformed.append(token.text_with_ws)
        return "".join(transformed).strip()
    
    def check(self, word):
        '''
        Used at Debug mode
        Could query CEFR analysis of specific word
        '''
        word = word.lower()
        print(f"\n{'='*15} CEFR Analysis: '{word}' {'='*15}")
        
        if word not in self.word_cefr_freq:
            print(f"Result: Word '{word}' not found.")
            return

        freq = self.word_cefr_freq[word]
        freqs = np.array([freq[cefr] for cefr in CEFR_LEVELS], dtype=float)
        total = freqs.sum()
        
        distributions = freqs / total
        weights = self.power_normalize(distributions)
        penalty = self.entropy_penalty(distributions)
        score = self.cefr_scores.get(word, 0.0)
        
        print(f"{'Level':<8} | {'Freq':<6} | {'Pct %':<8} | {'Power Weight %'}")
        print("-" * 55)
        for i, cefr in enumerate(CEFR_LEVELS):
            print(f"{cefr:<8} | {int(freqs[i]):<6} | {distributions[i]*100:<7.2f}% | {weights[i]*100:<15.2f}%")
        print("-" * 55)
        print(f"Penalty: {penalty:.4f}")
        print(f"Final Score: {score:.4f}")

# ==================== Main ====================
manager = None

def transform_sentence(sentence, source_level, target_level):
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")
    if source_level == target_level:
        return sentence

    global manager
    if manager is None: manager = CEFRManager("data.csv")
    result = manager.transform(sentence, source_level, target_level)
    
    if DEBUG_MODE:
        while True:
            user_word = input("\n[Check Mode] Word (or 'end'): ").strip().lower()
            if user_word == 'end': break
            if user_word: manager.check(user_word)
    
    return result