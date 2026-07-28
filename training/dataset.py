import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config.config import MODEL_NAME, MAX_SEQ_LEN


class LogDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        log = str(self.data.iloc[idx]["Text"])

        label = int(self.data.iloc[idx]["Label"])

        encoding = self.tokenizer(
            log,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }