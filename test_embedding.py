import unittest
from collections import Counter
import torch.nn as nn
import torch
import sys
import pandas
import numpy

def allocate_gpu():
    T = torch.ones(48 * 1000000, dtype=torch.float, device="cuda:0")
    in_bytes = T.element_size() * T.nelement()
    print (T.max(), " -> size = ", in_bytes / 1024 / 1024, "Mb")



class TestEmbedding(unittest.TestCase):

    def test_embedding(self):
        # Let's say you have 2 sentences(lowercased, punctuations removed) :
        sentences = "i am new to PyTorch i am having fun"

        words = sentences.split(' ')
        vocab = Counter(words) # create a dictionary
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        vocab_size = len(vocab)

        # map words to unique indices
        word2idx = {word: ind for ind, word in enumerate(vocab)}
        # word2idx = {'i': 0, 'am': 1, 'new': 2, 'to': 3, 'pytorch': 4, 'having': 5, 'fun': 6}
        encoded_sentences = [word2idx[word] for word in words]
        # encoded_sentences = [0, 1, 2, 3, 4, 0, 1, 5, 6]

        # let's say you want embedding dimension to be 3
        emb_dim = 3
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        word_vectors = emb_layer(torch.LongTensor(encoded_sentences))
        pass


    def test_load_pretrained_embedding(self):
        file_path = "C:/Users/User/Downloads/glove.6B/glove.6B.50d.txt"
        vocab,embeddings = [],[]
        with open(file_path, 'rt') as fi:
            full_content = fi.read().strip().split('\n')

        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)

        if '' in vocab:
            vocab.remove('')

        vals = list(numpy.arange(4, 4 + len(vocab)))
        word2idx = dict(zip(vocab, vals))

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

        # Load embedding weights into embedding layer
        embedding_tensor = torch.FloatTensor(embeddings)
        embedding_layer = nn.Embedding.from_pretrained(embedding_tensor)

        pass
