import unittest
import torch
import random


class TestMisc(unittest.TestCase):

    def test_random_indicies(self):
        x = list(range(100))
        xr = random.sample(x, 5)
        pass
