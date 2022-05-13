import stat
import numpy
from collections import Counter
import re
import torchtext


class TokenizerLanguageModel:
    def __init__(
        self,
        pad_token: str,
        start_token: str,
        end_token: str,
        unk_token: str,
        pad_token_num: int,
        start_token_num: int,
        end_token_num: int,
        unk_token_num: int,
    ):
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token

        self.pad_token_num = pad_token_num
        self.start_token_num = start_token_num
        self.end_token_num = end_token_num
        self.unk_token_num = unk_token_num

        self.word2idx = None
        self.word2idx_size = None
        self.idx2word = None

        return

    @staticmethod
    def cleanup(
        data: str,
        tokenizer,
    ):
        return tokenizer(data)

    @staticmethod
    def buildup(
        data: list,
    ):
        if len(data) == 0:
            raise ValueError('sequence is empty')

        out = ''
        for el in data:
            out += el + ' '
        out = out[:-1]

        return out

    def assemble_vocab(
        self,
        data: list,
    ):
        out = Counter(data)
        out = sorted(out, key=out.get, reverse=True)

        self.word2idx = {word: ind + 4 for ind, word in enumerate(out)}
        self.word2idx[self.pad_token] = self.pad_token_num
        self.word2idx[self.start_token] = self.start_token_num
        self.word2idx[self.end_token] = self.end_token_num
        self.word2idx[self.unk_token] = self.unk_token_num

        self.idx2word = {self.word2idx[key]: key for key in self.word2idx.keys()}

        self.word2idx_size = len(self.word2idx)

        return


    def load_vocab_from_file(self, filepath):
        """ Loads and assembles vocab from text file with unique word on each line """
        f = open(filepath, "r", encoding="utf-8")
        keys = f.read().split("\n")
        f.close()

        if '' in keys:
            keys.remove('')

        vals = list(numpy.arange(4, 4 + len(keys)))
        word2idx = dict(zip(keys, vals))

        word2idx["<pad>"] = 0
        word2idx["<sos>"] = 1
        word2idx["<eos>"] = 2
        word2idx["<unk>"] = 3

        # must pass, otherwise can cause embedding error
        assert max(word2idx.values()) < len(word2idx.values())

        # create opposite map idx2word
        keys = list(word2idx.keys())
        vals = list(word2idx.values())
        idx2word = dict(zip(vals, keys))

        self.word2idx = word2idx
        self.word2idx_size = len(word2idx)
        self.idx2word = idx2word


    def encode_seq(
        self,
        data: list,
    ):
        if self.word2idx is None:
            raise RuntimeError('vocab is empty')

        out = []
        for el in data:
            if el in self.word2idx.keys():
                out.append(self.word2idx[el])
            else:
                out.append(self.unk_token_num)

        return out

    def decode_seq(
        self,
        data: list,
    ):
        if self.idx2word is None:
            raise RuntimeError('vocab is empty')

        out = []
        for el in data:
            out.append(self.idx2word[el])

        return out
    
    def set_vocab(
        self,
        word2idx: dict,
    ):
        self.word2idx = word2idx.copy()
        
        self.idx2word = {self.word2idx[key]: key for key in self.word2idx.keys()}

        self.word2idx_size = len(self.word2idx)

        return

class TokenizerCollection:
    def __init__(self):
        return

    @staticmethod
    def basic_english_by_word(
        data: str,
    ):
        out = torchtext.data.utils.get_tokenizer('basic_english')(data)
        out = [el for el in out if el.isalpha()]
        return out

