import numpy as np
from torch.utils.data import Dataset


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
            seq = self.data[idx * self.sequence_length :] + [self.end_token_num] + [self.pad_token_num] * ((idx + 1) * self.sequence_length - self.data_length)
            return np.array([self.pad_token_num] + seq)
        
        seq = self.data[idx * self.sequence_length : (idx + 1) * self.sequence_length]
        return np.array([self.start_token_num] + seq + [self.end_token_num])


class DatasetClassifierModel(Dataset):
    def __init__(
            self,
            data_matrix: np.array,      # numpy 2d array: rows = num of reviews, cols = max review len
                                        # each row is an encoded sequence with <sos> at start,
                                        # <eos> at the end, with <pad> tokens finishing the sequence
            tgt_classes: np.array,      # numpy 1d array with class index 0 or 1
            # start_token: str,
            # end_token: str,
            # pad_token: str,
            # vocab: dict,
    ):
        assert data_matrix.shape[0] == tgt_classes.shape[0]
        self.data_length = data_matrix.shape[0]
        self.data_matrix = data_matrix
        self.tgt_classes = tgt_classes
        # self.start_token_num = vocab[start_token]
        # self.end_token_num = vocab[end_token]
        # self.pad_token_num = vocab[pad_token]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        """ Returns a tuple (encoded sequence, tgt_class) """
        return self.data_matrix[idx, :],\
               self.tgt_classes[idx]


        # if (idx + 1) * self.sequence_length > self.data_length:
        #     seq = self.data[idx * self.sequence_length:] + [self.end_token_num] + [self.pad_token_num] * (
        #                 (idx + 1) * self.sequence_length - self.data_length)
        #     return np.array([self.pad_token_num] + seq)
        #
        # seq = self.data[idx * self.sequence_length: (idx + 1) * self.sequence_length]
        # return np.array([self.start_token_num] + seq + [self.end_token_num])

