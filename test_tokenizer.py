import unittest
import numpy
from webencodings import encode

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
        tokenizer.load_vocab_from_file("vocabs/20k.txt")

        some_random_string = "Hello, my name is HdsUAHUDAWYDT"
        seq = tokenizer.cleanup(data=some_random_string, tokenizer=TokenizerCollection.basic_english_by_word)

        encoded_seq = tokenizer.encode_seq(seq)
        decoded_seq = tokenizer.decode_seq(encoded_seq)
        pass


    def test_load_pretrained_embedding(self):
        tokenizer = self.create_tokenizer()
        embedding_weights = tokenizer.load_pretrained_embedding("C:/Users/User/Downloads/glove.6B/glove.6B.50d.txt")
        pass


    def test_IMBD_dataset_usage_example(self):
        """ This is for classification dataset """
        tokenizer = self.create_tokenizer()
        embedding_weights = tokenizer.load_pretrained_embedding("pretrained_embedding_vocab/glove.6B.50d.top30K.txt")
        df = pandas.read_csv("data/classification/IMDB_dataset.csv")

        vocab = tokenizer.word2idx

        # encoded_reviews = [0] * df.shape[0]
        # tgt_classes = [0] * df.shape[0]

        max_review_len = 0

        start_token = vocab[model_constants.start_token]
        end_token   = vocab[model_constants.end_token]
        pad_token   = vocab[model_constants.pad_token]
        unk_token   = vocab[model_constants.unk_token]

        for i in range(df.shape[0]):
            review = df.iloc[i, 0]
            seq = tokenizer.cleanup(data=review, tokenizer=TokenizerCollection.basic_english_by_word)
            if len(seq) > max_review_len:
                max_review_len = len(seq)

            enc_seq = tokenizer.encode_seq(seq)

            # surround sequence with start, end tokens:
            enc_seq.insert(0, start_token)
            enc_seq.append(end_token)

            tgt_class = 1 if df.iloc[i, 1] == "positive" else 0
            pass

        # Create matrix filled with padding tokens:
        data = [0] * df.shape[0]  # this will be a list of tuples (encoded_review, 0 or 1)
        data_matrix = numpy.zeros(shape=(df.shape[0], max_review_len)) + pad_token


