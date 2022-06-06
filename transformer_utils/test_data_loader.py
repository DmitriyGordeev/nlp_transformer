import unittest
from data_loader import *
from tokenizer import TokenizerLanguageModel, TokenizerCollection
import model_constants
from torch.utils.data import DataLoader
import pandas
import numpy


def prepare_data(filepath: str, tokenizer: TokenizerLanguageModel):
    df = pandas.read_csv(filepath)
    vocab = tokenizer.word2idx
    max_review_len = 0

    start_token = vocab[model_constants.start_token]
    end_token = vocab[model_constants.end_token]
    pad_token = vocab[model_constants.pad_token]

    encoded_sequences = []
    tgt_classes = numpy.zeros(df.shape[0])

    for i in range(df.shape[0]):
        review = df.iloc[i, 0]
        seq = tokenizer.cleanup(data=review, tokenizer=TokenizerCollection.basic_english_by_word)
        if len(seq) > max_review_len:
            max_review_len = len(seq)

        enc_seq = tokenizer.encode_seq(seq)

        # surround sequence with start, end tokens:
        enc_seq.insert(0, start_token)
        enc_seq.append(end_token)
        encoded_sequences.append(numpy.array(enc_seq))       # todo: optimize this

        tgt_class = 1 if df.iloc[i, 1] == "positive" else 0
        tgt_classes[i] = tgt_class

    # Create matrix filled with padding tokens:
    data_matrix = numpy.zeros(shape=(df.shape[0], max_review_len + 2)) + pad_token      # +2 because we have sos and eos
    for i in range(len(encoded_sequences)):
        data_matrix[i, :len(encoded_sequences[i])] = encoded_sequences[i]
    return data_matrix, tgt_classes



class TestDataLoader(unittest.TestCase):

    def test_dataloader_example_usage(self):

        # Create tokenizer instance
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

        # Some simple text
        text = "A brown dog jumped on the fox"

        # Clean and assemble vocab from the text
        text = tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        tokenizer.assemble_vocab(text)

        encoded_seq = tokenizer.encode_seq(text)

        # Dataset and DataLoader example usages:
        train_dataset = DatasetLanguageModel(
            data=encoded_seq,
            sequence_length=2,                          # some arbitrary length to test
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            pad_token=model_constants.pad_token,
            vocab=tokenizer.word2idx,
        )

        dataloader_example = DataLoader(
            dataset=train_dataset,
            batch_size=1,       # 1 batch = 1 sentence
            shuffle=True,
        )

        for batch in dataloader_example:
            list_of_wordidx = list(batch[0, :].numpy())     # converts 1st sample in a batch into array of indexes (each index -> word)
            print (tokenizer.decode_seq(list_of_wordidx))   # decode and print sentence to show the example how DataLoader and Dataset work


    def test_dataloader_classifier_example_usage(self):
        # Create tokenizer instance
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
        embedding_weights = tokenizer.load_pretrained_embedding("pretrained_embedding_vocab/glove.6B.50d.top30K.txt")
        data_matrix, tgt_classes = prepare_data("data/classification/IMDB_dataset.csv", tokenizer)
        pass


    def test_split_into_tables(self):
        df = pandas.read_csv("data/classification/IMDB_dataset.csv")
        train_portion = 0.7

        train_size = int(df.shape[0] * train_portion)
        df_train = df.head(train_size)
        df_train.to_csv("data/classification/train.csv", index=False)

        val_size = int((df.shape[0] - train_size) / 2)
        df_val = df.iloc[train_size : train_size + val_size, :]
        df_val.to_csv("data/classification/val.csv", index=False)

        test_size = df.shape[0] - train_size - val_size
        df_test = df.tail(test_size)
        df_test.to_csv("data/classification/test.csv", index=False)









