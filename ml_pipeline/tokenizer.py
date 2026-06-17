import json
import os
from typing import List, Dict

class WordTokenizer:
    """
    Word-level tokenizer designed for offline execution.
    Features out-of-vocabulary (OOV) handling and padding.
    """
    def __init__(self, max_seq_length: int = 20):
        self.max_seq_length = max_seq_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
    def fit(self, texts: List[str]):
        """Builds vocabulary from a list of strings."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a padded integer sequence."""
        words = text.lower().split()
        seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        
        # Pad or truncate
        if len(seq) < self.max_seq_length:
            seq += [self.word2idx["<PAD>"]] * (self.max_seq_length - len(seq))
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def save(self, path: str):
        """Saves the vocabulary for ONNX/Inference consumption."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"word2idx": self.word2idx, "max_seq_length": self.max_seq_length}, f)
            
    def load(self, path: str):
        """Loads vocabulary."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.vocab_size = len(self.word2idx)
            self.max_seq_length = data["max_seq_length"]
