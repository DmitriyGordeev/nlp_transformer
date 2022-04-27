import unittest
from collections import Counter
import re
import numpy
import torch
import torch.nn as nn

from tokenizer import Tokenizer


class TestTextConverter(unittest.TestCase):

    def test_text(self):
        with open("space.txt", "r") as f:
            text = f.read()
            tp = Tokenizer()
            tp.assemble_vocab(text)

            print("australian", tp.encode_sentence("australian"))
            print(tp.decode_sentence([1, 3, 4, 2]))


    def test_cross_entropy(self):
        # Example of target with class indices
        loss = nn.CrossEntropyLoss()
        inp = torch.randn(1, 5, requires_grad=True)
        target = torch.empty(1, dtype=torch.long).random_(5)
        output = loss(inp, target)
        output.backward()


    def test_stepping(self):
        cursor = 0
        n_src = 3
        n_tgt = 3
        shift = 4

        arr = numpy.arange(20)

        data = list()
        while cursor < len(arr) - (n_src + n_tgt):
            src = arr[cursor:cursor + n_src]
            tgt = arr[cursor + n_src : cursor + n_src + n_tgt]
            data.append([src, tgt])
            cursor += shift
            pass
        pass
