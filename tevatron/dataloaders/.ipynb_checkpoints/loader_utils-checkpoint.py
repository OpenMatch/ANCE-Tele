import datasets
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    BatchEncoding, 
    DataCollatorWithPadding
)
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(
        self, 
        dataset: datasets.Dataset, 
        tokenizer: PreTrainedTokenizer, 
        max_len=128
    ):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text
    

@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features
    
    
    
