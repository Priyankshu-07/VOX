import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict
from tokenizer import WordTokenizer
INTENT_MAP = {
    "get_address": 0,
    "report_delay": 1,
    "order_issue": 2,
    "customer_unavailable": 3,
    "navigation_help": 4
}
REV_INTENT_MAP = {v: k for k, v in INTENT_MAP.items()}
class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: WordTokenizer, is_train: bool = False):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        if is_train:
            self.tokenizer.fit(self.data['text'].tolist()) 
        self.texts = self.data['text'].tolist()
        self.labels = [INTENT_MAP[intent] for intent in self.data['intent'].tolist()]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_seq = self.tokenizer.encode(text)
        return {
            "input_ids": torch.tensor(encoded_seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
