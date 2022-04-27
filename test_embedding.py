import unittest
from collections import Counter
import torch.nn as nn
import torch

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


    def test_shifting(self):
        sentences = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]

        T = torch.FloatTensor(sentences)
        pass