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
            seq = self.data[idx * self.sequence_length:] + [self.end_token_num] + [self.pad_token_num] * (
                        (idx + 1) * self.sequence_length - self.data_length)
            return np.array([self.start_token_num] + seq)

        seq = self.data[idx * self.sequence_length: (idx + 1) * self.sequence_length]
        return np.array([self.start_token_num] + seq + [self.end_token_num])


class DatasetSummarizerModel(Dataset):
    def __init__(
            self,
            data: list,
            src_max_len: int,
            tgt_max_len: int,
            start_token: str,
            end_token: str,
            pad_token: str,
            vocab: dict,
    ):
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.data = data.copy()

        self.start_token_num = vocab[start_token]
        self.end_token_num = vocab[end_token]
        self.pad_token_num = vocab[pad_token]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        src = self.data[0][idx]
        tgt = self.data[1][idx]

        src_padded = np.array([self.start_token_num] + src + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.src_max_len + 2 - len(
                                                                                                                   src)))
        tgt_padded = np.array([self.start_token_num] + tgt + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.tgt_max_len + 2 - len(
                                                                                                                   tgt)))

        return [src_padded, tgt_padded]


class DatasetSummarizerBillSumv3(Dataset):
    def __init__(self, tuples,
                 src_max_len: int,
                 tgt_max_len: int,
                 start_token: str,
                 end_token: str,
                 pad_token: str,
                 vocab):
        self.tuples = tuples  # each element is a tuple (text, summary)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        self.start_token_num = vocab[start_token]
        self.end_token_num = vocab[end_token]
        self.pad_token_num = vocab[pad_token]

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, index):
        src = self.tuples[index][0]
        tgt = self.tuples[index][1]

        src_padded = np.array([self.start_token_num] + src + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.src_max_len - len(
                                                                                                                   src)))
        tgt_padded = np.array([self.start_token_num] + tgt + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.tgt_max_len - len(
                                                                                                                   tgt)))
        return [src_padded, tgt_padded]



class DatasetSummarizerBPE(Dataset):
    def __init__(self, text_summary_tuples,
                 src_max_len,
                 tgt_max_len,
                 start_token_num,
                 end_token_num,
                 pad_token_num):

        self.tuples = text_summary_tuples  # each element is a tuple (text, summary)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.start_token_num = start_token_num
        self.end_token_num = end_token_num
        self.pad_token_num = pad_token_num


    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, index):
        src = self.tuples[index][0]
        tgt = self.tuples[index][1]

        src_padded = np.array([self.start_token_num] + src + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.src_max_len - len(
                                                                                                                   src)))
        tgt_padded = np.array([self.start_token_num] + tgt + [self.end_token_num] + [self.pad_token_num] * max(0,
                                                                                                               self.tgt_max_len - len(
                                                                                                                   tgt)))
        return [src_padded, tgt_padded]



class DatasetClassifierModel(Dataset):
    def __init__(
            self,
            data_matrix: np.array,  # numpy 2d array: rows = num of reviews, cols = max review len
            # each row is an encoded sequence with <sos> at start,
            # <eos> at the end, with <pad> tokens finishing the sequence
            tgt_classes: np.array  # numpy 1d array with class index 0 or 1
    ):
        assert data_matrix.shape[0] == tgt_classes.shape[0]
        self.data_length = data_matrix.shape[0]
        self.data_matrix = data_matrix
        self.tgt_classes = tgt_classes

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        """ Returns a tuple (encoded sequence, tgt_class) """
        return self.data_matrix[idx, :], \
               self.tgt_classes[idx]
