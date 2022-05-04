import unittest
from data_loader import *
from tokenizer import TokenizerLanguageModel, TokenizerCollection
import model_constants
from torch.utils.data import DataLoader



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
