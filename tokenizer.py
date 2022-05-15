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


    def load_pretrained_embedding(self, filepath, top_n=-1):
        """ Loads pretrained embedding and respective vocab
            @:param filepath - path to file with words and pretrained embedding weights
            @:param top_n - first n words from pretrained model (by default -1 - means all)
            @:return embedding weights as list of vectors (each vector is of dimension size)
         """
        vocab, embeddings = [], []
        with open(filepath, 'rt') as fi:
            full_content = fi.read().strip().split('\n')

        vocab_size = min(top_n, len(full_content))
        if vocab_size < 0:
            vocab_size = len(full_content)

        for i in range(vocab_size):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)            # todo: replace append() with [idx]
            embeddings.append(i_embeddings)

        assert len(embeddings) > 0
        embedding_dim = len(embeddings[0])

        if '' in vocab:
            vocab.remove('')

        vals = list(numpy.arange(4, 4 + len(vocab)))
        word2idx = dict(zip(vocab, vals))

        word2idx["<pad>"] = 0
        word2idx["<sos>"] = 1
        word2idx["<eos>"] = 2
        word2idx["<unk>"] = 3

        # add extra embedding vectors for every special token (fill with zeros?)
        # TODO: which weights should we use for <eos> and <unk> ???
        embeddings.insert(0, [0] * embedding_dim)
        embeddings.insert(1, [0] * embedding_dim)
        embeddings.insert(2, [0] * embedding_dim)
        embeddings.insert(3, [0] * embedding_dim)

        # must pass, otherwise can cause embedding error
        assert max(word2idx.values()) < len(word2idx.values())
        assert len(embeddings) == len(word2idx)

        # create opposite map idx2word
        keys = list(word2idx.keys())
        vals = list(word2idx.values())
        idx2word = dict(zip(vals, keys))

        self.word2idx = word2idx
        self.word2idx_size = len(word2idx)
        self.idx2word = idx2word
        return embeddings



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

