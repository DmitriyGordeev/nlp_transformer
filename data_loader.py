from torch.utils.data import Dataset

import numpy as np


class DatasetLanguageModel(Dataset):
    def __init__(
        self,
        data: list,
        sequence_length: int,
        start_token: str,
        end_token: str,
        pad_token: str,
        vocab: dict,
    ):
        self.data_length = len(data)
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.start_token_num = vocab[start_token]
        self.end_token_num = vocab[end_token]
        self.pad_token_num = vocab[pad_token]
    
    def __len__(self):

        if self.data_length % self.sequence_length == 0:
            return self.data_length // self.sequence_length

        return self.data_length // self.sequence_length + 1

    def __getitem__(self, idx):

        if (idx + 1) * self.sequence_length > self.data_length:
            seq = self.data[idx * self.sequence_length :] + [self.end_token_num] +[self.pad_token_num] * ((idx + 1) * self.sequence_length - self.data_length)
            return np.array([self.pad_token_num] + seq)
        
        seq = self.data[idx * self.sequence_length : (idx + 1) * self.sequence_length]
        return np.array([self.start_token_num] + seq + [self.end_token_num])