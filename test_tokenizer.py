import unittest
import numpy
import model_constants
from tokenizer import TokenizerLanguageModel, TokenizerCollection
import pandas


class TestTokenizer(unittest.TestCase):

    @staticmethod
    def create_tokenizer():
        tokenizer = TokenizerLanguageModel(
            pad_token=model_constants.pad_token,
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            unk_token=model_constants.unk_token,
            pad_token_num=model_constants.pad_token_num,
            start_token_num=model_constants.start_token_num,
            end_token_num=model_constants.end_token_num,
            unk_token_num=model_constants.unk_token_num
        )
        return tokenizer


    def test_short_text(self):
        text = "A brown dog jumped on the fox"

        tokenizer = self.create_tokenizer()
        text = tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        tokenizer.assemble_vocab(text)

        encoded_seq = tokenizer.encode_seq(text)
        decoded_seq = tokenizer.decode_seq(encoded_seq)
        pass


    def test_text(self):
        tokenizer = self.create_tokenizer()

        with open("data/space.txt", "r", encoding="utf-8") as f:
            text = f.read()
            text = tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
            tokenizer.assemble_vocab(text)
            print(tokenizer.encode_seq(["pluto"]))


    def test_sampling(self):
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


    def test_encode_decode_with_loaded_vocab(self):
        tokenizer = self.create_tokenizer()
        tokenizer.load_vocab_from_file("C:/Users/Администратор/Downloads/high-frequency-vocabulary-master/20k.txt")

        some_random_string = "Hello, my name is HdsUAHUDAWYDT"
        seq = tokenizer.cleanup(data=some_random_string, tokenizer=TokenizerCollection.basic_english_by_word)

        encoded_seq = tokenizer.encode_seq(seq)
        decoded_seq = tokenizer.decode_seq(encoded_seq)

        pass
