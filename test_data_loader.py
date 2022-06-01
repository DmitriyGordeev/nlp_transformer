import unittest
from data_loader import *
from tokenizer import TokenizerLanguageModel, TokenizerCollection
import model_constants
from torch.utils.data import DataLoader
import pandas
import numpy



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
        df = pandas.read_csv("data/classification/IMDB_dataset.csv")

        vocab = tokenizer.word2idx
        max_review_len = 0

        start_token = vocab[model_constants.start_token]
        end_token = vocab[model_constants.end_token]
        pad_token = vocab[model_constants.pad_token]
        unk_token = vocab[model_constants.unk_token]

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

            tgt_class = 1 if df.iloc[i, 1] == "positive" else 0
            tgt_classes[i] = tgt_class

        # Create matrix filled with padding tokens:
        data = [0] * df.shape[0]  # this will be a list of tuples (encoded_review, 0 or 1)
        data_matrix = numpy.zeros(shape=(df.shape[0], max_review_len)) + pad_token

        classifierDL = DatasetClassifierModel(
            data_matrix=data_matrix,
            tgt_classes=tgt_classes
            # start_token=start_token,
            # end_token=end_token,
            # pad_token=pad_token,
            # vocab=vocab
        )

        dataloader_train = DataLoader(
            dataset=classifierDL,
            batch_size=8,
            shuffle=True,
        )

        for batch in dataloader_train:
            # batch is a list of 2 tensors
            # batch[0] - tensor(batch_size, max_review_len)
            # batch[1] - tensor(batch_size) classes
            pass



        # TODO: how to output as a tensor ?
        

