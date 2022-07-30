import torch
import torch.nn as nn
import torch.optim as top
import matplotlib
import jsonlines
import numpy as np
from pathlib import Path

from transformer_utils import training_setup_abstract
from transformer_utils import model
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection

# from transformer_utils.model_constants import *
from transformer_utils.data_loader import DatasetSummarizerBPE
from datasets import load_dataset
matplotlib.use("Agg")

import os
import random
from bpemb import BPEmb


class ModelParams(training_setup_abstract.ModelParams):
    pass


class TrainParams(training_setup_abstract.TrainParams):
    pass


class SummarizerSetup(training_setup_abstract.TrainingSetup):

    def __init__(self,
                 is_gpu: bool,
                 is_resume_mode: bool,
                 train_params: TrainParams,
                 model_params: ModelParams):
        super(SummarizerSetup, self).__init__(is_gpu=is_gpu,
                                              is_resume_mode=is_resume_mode,
                                              train_params=train_params,
                                              model_params=model_params)

        self.bpe_embedding_dim = 50
        self.bpemb_en = BPEmb(lang="en", dim=self.bpe_embedding_dim, cache_dir=Path("bpe_model/"))
        self.pad_token_index = None

    def load_billsumv3(self, filepath: str):
        """ :param filepath - path to *.jsonl from BillSumv3 dataset """
        data_tuples = []
        max_text_len = 0
        max_summary_len = 0
        with jsonlines.open(filepath) as f:
            print(f"Encoding text and summary strings ...")
            for line in f.iter():
                text = line['text']
                summary = line['summary']
                title = line['title']

                # THIS IS FOR TEST ONLY:
                text = text[:400]
                summary = summary[:100]

                # encode text and summary with bpemb
                text_ids = self.bpemb_en.encode_ids(text)
                summary_ids = self.bpemb_en.encode_ids(summary)

                data_tuples.append((text_ids, summary_ids))
                if max_text_len < len(text_ids):
                    max_text_len = len(text_ids)
                if max_summary_len < len(summary_ids):
                    max_summary_len = len(summary_ids)

        return data_tuples, max_text_len, max_summary_len


    def load_multi_news(self, key: str):
        """ :param key should be one of 'train', 'validation', 'test' """
        dataset = load_dataset("multi_news")
        container = dataset[key]
        data_tuples = []
        max_text_len = 0
        max_summary_len = 0
        for element in container:
            document = element["document"]
            summary = element['summary']

            # THIS IS FOR TEST ONLY:
            document = document[:400]
            summary = summary[:100]

            # encode text and summary with bpemb
            text_ids = self.bpemb_en.encode_ids(document)
            summary_ids = self.bpemb_en.encode_ids(summary)

            data_tuples.append((text_ids, summary_ids))
            if max_text_len < len(text_ids):
                max_text_len = len(text_ids)
            if max_summary_len < len(summary_ids):
                max_summary_len = len(summary_ids)
        return data_tuples, max_text_len, max_summary_len



    def load_data(self,
                  train_path: str,
                  test_path: str,
                  val_path: str):

        # 1. Resampling params (todo: move to other params)
        self.resampling_portion = 0.1
        self.resampling_freq_epochs = 16

        # 2. Load embedding vectors
        embedding_vectors = self.bpemb_en.vectors

        # add padding token and respective random vector to the embedding,
        # because 'bpemb_en' doesn't have <pad> by default
        random_pad_vector = np.random.random((1, self.bpe_embedding_dim))
        embedding_vectors = np.append(embedding_vectors, random_pad_vector, axis=0)
        self.pad_token_index = embedding_vectors.shape[0] - 1

        self.pretrained_embedding = torch.FloatTensor(embedding_vectors)

        # 4. save vocab to vocab.pt file
        key_to_index = self.bpemb_en.emb.key_to_index
        key_to_index["<pad>"] = self.pad_token_index  # add as the last element in the vocab
        # if vocab size = 10001, index of <pad> is 10000
        self.word2idx = key_to_index
        self.word2idx_size = len(self.word2idx)
        torch.save(self.word2idx, "models/" + self.train_params.path_nm + "/vocab.pt")

        # 5. Load data:
        # Train file -> self.train_data as tuple (array, max_text_len, max_summary_len)
        if os.path.isfile("cached_data_train.pt"):  # load cached data to avoid encoding again
            train_tuples, max_text_len, max_summary_len = torch.load("cached_data_train.pt")
            print(f"loaded cached train data, "
                  f"len(train_tuples) = {len(train_tuples)}, "
                  f"max_text_len = {max_text_len}, "
                  f"max_summary_len = {max_summary_len}")
        else:
            # train_tuples, max_text_len, max_summary_len = self.load_billsumv3(train_path)
            train_tuples, max_text_len, max_summary_len = self.load_multi_news("train")
            torch.save((train_tuples, max_text_len, max_summary_len), f"cached_data_train.pt")

        print(f"num train pairs (text,summary) = {len(train_tuples)}")
        self.train_data = (train_tuples, max_text_len, max_summary_len)
        del train_tuples  # cleanup memory

        # Validation file -> self.val_data as tuple (array, max_text_len, max_summary_len)
        if os.path.isfile("cached_data_val.pt"):  # load cached data to avoid encoding again
            val_tuples, max_text_len, max_summary_len = torch.load("cached_data_val.pt")
            print(f"loaded cached validation data, "
                  f"len(val_tuples) = {len(val_tuples)}, "
                  f"max_text_len = {max_text_len}, "
                  f"max_summary_len = {max_summary_len}")
        else:
            # val_tuples, max_text_len, max_summary_len = self.load_billsumv3(val_path)
            val_tuples, max_text_len, max_summary_len = self.load_multi_news("validation")
            torch.save((val_tuples, max_text_len, max_summary_len), "cached_data_val.pt")

        print(f"num val pairs (text,summary) = {len(val_tuples)}")
        self.val_dataset = DatasetSummarizerBPE(val_tuples,
                                                max_text_len,
                                                max_summary_len,
                                                start_token_num=self.bpemb_en.BOS,
                                                end_token_num=self.bpemb_en.EOS,
                                                pad_token_num=self.pad_token_index)

        del val_tuples  # cleanup memory

        # Test file -> test dataset
        if os.path.isfile("cached_data_test.pt"):  # load cached data to avoid encoding again
            test_tuples, max_text_len, max_summary_len = torch.load("cached_data_test.pt")
            print(f"loaded cached test data, "
                  f"len(test_tuples) = {len(test_tuples)}, "
                  f"max_text_len = {max_text_len}, "
                  f"max_summary_len = {max_summary_len}")
        else:
            # test_tuples, max_text_len, max_summary_len = self.load_billsumv3(test_path)
            test_tuples, max_text_len, max_summary_len = self.load_multi_news("test")
            torch.save((test_tuples, max_text_len, max_summary_len), "cached_data_test.pt")

        print(f"num test pairs (text,summary) = {len(test_tuples)}")
        self.test_dataset = DatasetSummarizerBPE(test_tuples,
                                                 max_text_len,
                                                 max_summary_len,
                                                 start_token_num=self.bpemb_en.BOS,
                                                 end_token_num=self.bpemb_en.EOS,
                                                 pad_token_num=self.pad_token_index)
        del test_tuples  # cleanup memory

    def setup_nn(self):
        self.nn_model = model.TransformerLanguageModel(
            num_tokens=self.word2idx_size,
            d_model=self.model_params.d_model,
            nhead=self.model_params.nhead,
            num_encoder_layers=self.model_params.num_encoder_layers,
            num_decoder_layers=self.model_params.num_decoder_layers,
            dim_feedforward=self.model_params.dim_feedforward,
            dropout_p=self.model_params.dropout_p,
        )
        self.nn_model.load_and_freeze_pretrained_embedding(self.pretrained_embedding)

    def nn_forward(self, batch, print_enabled=False):
        """ Helper function to be invoked everywhere on training, validation and test stages
        :param batch:
        :param print_enabled: if true prints predicted sequence
        :return: loss
        """
        src = batch[0].to(self.device)
        tgt_input = batch[1][:, :-1].to(self.device)
        tgt_expected = batch[1][:, 1:].to(self.device)

        # Get mask to mask out the next words
        sequence_length = tgt_input.size(1)
        tgt_mask = self.nn_model.get_tgt_mask(sequence_length).to(self.device)
        # src_key_padding_mask = self.nn_model.create_pad_mask(src, self.tokenizer.pad_token_num)
        # tgt_key_padding_mask = self.nn_model.create_pad_mask(tgt_input, self.tokenizer.pad_token_num)

        src_key_padding_mask = self.nn_model.create_pad_mask(src, self.pad_token_index)
        tgt_key_padding_mask = self.nn_model.create_pad_mask(tgt_input, self.pad_token_index)

        # Standard training except we pass in y_input and tgt_mask
        pred = self.nn_model(
            src=src,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        tgt_expected = tgt_expected.type(torch.int64)

        loss = self.criterion(pred, tgt_expected)

        if print_enabled:
            # Print predicted sequence
            predicted_sequence = self.predict(src[0:1, :], max_length=self.train_params.inference_max_len)

            # TODO: move from stdout to the logger

            # decode src, tgt and prediction into human-readable string
            print("=================================================================")
            print(f"Predicted sequence, max_inference_len = {self.train_params.inference_max_len} : ")
            # print(f"src  = {' '.join(self.tokenizer.decode_seq(src[0, :].view(-1).tolist()))}")
            # print(f"tgt  = {' '.join(self.tokenizer.decode_seq(tgt_expected[0, :].view(-1).tolist()))}")
            # print(f"pred = {' '.join(self.tokenizer.decode_seq(predicted_sequence))}")

            src_ids = src[0, :].view(-1).tolist()
            tgt_ids = tgt_expected[0, :].view(-1).tolist()
            pred_ids = predicted_sequence

            # remove pad token (because self.bpemb_en doesn't have it)
            src_ids = [x for x in src_ids if x != self.pad_token_index]
            tgt_ids = [x for x in tgt_ids if x != self.pad_token_index]
            pred_ids = [x for x in pred_ids if x != self.pad_token_index]

            print(f"src  = {self.bpemb_en.decode_ids(src_ids)}")
            print(f"tgt  = {self.bpemb_en.decode_ids(tgt_ids)}")
            print(f"pred = {self.bpemb_en.decode_ids(pred_ids)}")

            print("=================================================================")

        return pred, loss

    def predict(self, input_sequence, max_length=10):
        """ Infer sequence from input_sequence """
        self.nn_model.eval()
        y_input = torch.tensor([[self.bpemb_en.BOS]], dtype=torch.long, device=self.device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.nn_model.get_tgt_mask(y_input.size(1)).to(self.device)

            pred = self.nn_model(input_sequence, y_input, tgt_mask)

            next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
            next_item = torch.tensor([[next_item]], device=self.device)

            # ignore <pad> tokens
            if next_item == self.pad_token_index:
                continue

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.bpemb_en.EOS:
                break

        return y_input.view(-1).tolist()

    def setup_optimizers(self):
        self.optimizer = top.Adam(self.nn_model.parameters(),
                                  lr=self.train_params.learning_rate,
                                  weight_decay=self.train_params.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    patience=5,
                                                                    threshold=0.001,
                                                                    factor=0.5)

    def resampling(self):
        """ Overriding method from the parent class """
        if 0 < self.resampling_portion < 1:
            full_data_size = len(self.train_data[0])

            # how much data should be in a resampled chunk
            n_how_much = int(full_data_size * self.resampling_portion)
            subsample = random.sample(self.train_data[0], n_how_much)

            self.train_dataset = DatasetSummarizerBPE(subsample,
                                                      src_max_len=self.train_data[1],
                                                      tgt_max_len=self.train_data[2],
                                                      start_token_num=self.bpemb_en.BOS,
                                                      end_token_num=self.bpemb_en.EOS,
                                                      pad_token_num=self.pad_token_index)

        return super(SummarizerSetup, self).resampling()
