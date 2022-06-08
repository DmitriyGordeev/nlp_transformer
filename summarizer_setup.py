import torch
import torch.nn as nn
import torch.optim as top
import matplotlib
import jsonlines


from transformer_utils import training_setup_abstract
from transformer_utils import model
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection

from transformer_utils.model_constants import *
from transformer_utils.data_loader import DatasetSummarizerBillSumv3

matplotlib.use("Agg")

import os
import random


class ModelParams(training_setup_abstract.ModelParams):
    pass


class TrainParams(training_setup_abstract.TrainParams):
    pass


class SummarizerSetup(training_setup_abstract.TrainingSetup):

    # @staticmethod
    # def read_files(src_dir: str, tgt_dir: str):
    #
    #     # src_path_list = glob.glob('data/src/*.txt')
    #     src_path_list = glob.glob(src_dir + "/*.txt")
    #     src_list = []
    #     for el in src_path_list:
    #         f = open(el, 'r', encoding='utf-8')
    #         text = f.read()
    #         src_list.append(text)
    #         f.close()
    #
    #     # tgt_path_list = glob.glob('data/tgt/*.txt')
    #     tgt_path_list = glob.glob(tgt_dir + '/*.txt')
    #     tgt_list = []
    #     for el in tgt_path_list:
    #         f = open(el, 'r', encoding='utf-8')
    #         text = f.read()
    #         tgt_list.append(text)
    #         f.close()
    #
    #     if len(src_list) != len(tgt_list):
    #         raise RuntimeError('the number of src samples is not equal to the number of tgt samples')
    #
    #     return src_list, tgt_list
    #
    # @staticmethod
    # def train_val_test_split(
    #         src: list,
    #         tgt: list,
    #         ratio=(7, 2, 1),
    # ):
    #     seq_len = len(src)
    #     weight = []
    #     s = 0
    #     for el in (7, 2, 1):
    #         if el < 0:
    #             raise ValueError('ratio elements must be >= 0')
    #         s += el
    #         weight.append(s)
    #     weight = [int(w * seq_len / s) for w in weight]
    #
    #     idx = np.arange(seq_len)
    #     random.shuffle(idx)
    #
    #     train_idx = idx[:weight[0]]
    #     val_idx = idx[weight[0]:weight[1]]
    #     test_idx = idx[weight[1]:]
    #
    #     train_src = [src[i] for i in train_idx]
    #     val_src = [src[i] for i in val_idx]
    #     test_src = [src[i] for i in test_idx]
    #
    #     train_tgt = [tgt[i] for i in train_idx]
    #     val_tgt = [tgt[i] for i in val_idx]
    #     test_tgt = [tgt[i] for i in test_idx]
    #
    #     src_out = [train_src, val_src, test_src]
    #     tgt_out = [train_tgt, val_tgt, test_tgt]
    #
    #     return src_out, tgt_out
    #
    # def data_cleanup(
    #         self,
    #         src,
    #         tgt,
    #         tokenizer,
    # ):
    #     src_out = []
    #     tgt_out = []
    #
    #     src_max_len = -1
    #     tgt_max_len = -1
    #
    #     for i in range(len(src)):
    #         src_out_i = []
    #         tgt_out_i = []
    #
    #         for j in range(len(src[i])):
    #             src_out_i.append(self.tokenizer.cleanup(src[i][j], tokenizer))
    #             tgt_out_i.append(self.tokenizer.cleanup(tgt[i][j], tokenizer))
    #
    #             src_max_len = max(src_max_len, len(src_out_i[-1]))
    #             tgt_max_len = max(tgt_max_len, len(tgt_out_i[-1]))
    #
    #         src_out.append(src_out_i)
    #         tgt_out.append(tgt_out_i)
    #
    #     words = []
    #     for i in range(len(src_out[0])):
    #         for el in src_out[0][i]:
    #             words.append(el)
    #
    #     for i in range(len(tgt_out[0])):
    #         for el in tgt_out[0][i]:
    #             words.append(el)
    #
    #     return src_out, tgt_out, words, src_max_len, tgt_max_len

    def load_billsumv3(self, filepath: str) -> tuple:
        """ :param filepath - path to *.jsonl from BillSumv3 dataset
            :return list of tuples (text, summary) which are encoded with tokenizer already
        """
        data_tuples = []
        max_text_len = 0
        max_summary_len = 0
        with jsonlines.open(filepath) as f:
            print(f"Encoding text and summary strings ...")
            for line in f.iter():
                text = line['text']
                summary = line['summary']
                title = line['title']

                # encode text
                text = self.tokenizer.cleanup(text, tokenizer=TokenizerCollection.basic_english_by_word)
                text = self.tokenizer.encode_seq(text)

                # encode summary
                summary = self.tokenizer.cleanup(summary, tokenizer=TokenizerCollection.basic_english_by_word)
                summary = self.tokenizer.encode_seq(summary)

                # data_tuples will be passed to the Dataset
                data_tuples.append((text, summary))
                if max_text_len < len(text):
                    max_text_len = len(text)
                if max_summary_len < len(summary):
                    max_summary_len = len(summary)

        return data_tuples, max_text_len, max_summary_len


    def load_data(
            self,
            train_path: str,
            test_path: str,
            val_path: str,
    ):
        """ Reads file, tokenize and prepare tensors to train """
        self.tokenizer = TokenizerLanguageModel(
            pad_token=special_tokens['tokens']['pad_token'],
            start_token=special_tokens['tokens']['start_token'],
            end_token=special_tokens['tokens']['end_token'],
            unk_token=special_tokens['tokens']['unk_token'],
            pad_token_num=special_tokens['token_nums']['pad_token'],
            start_token_num=special_tokens['token_nums']['start_token'],
            end_token_num=special_tokens['token_nums']['end_token'],
            unk_token_num=special_tokens['token_nums']['unk_token'],
        )

        # TODO: expose these parameters to model_config
        self.resampling_portion = 0.1
        self.resampling_freq_epochs = 2

        # load glove embedding from file
        self.pretrained_embedding = self.tokenizer.load_pretrained_embedding(
            "pretrained_embedding_vocab/glove.6B.50d.top30K.txt",
            top_n=25000
        )
        self.word2idx = self.tokenizer.word2idx
        self.idx2word = self.tokenizer.idx2word
        self.word2idx_size = self.tokenizer.word2idx_size

        # Reading BillSumV3 dataset

        train_tuples = None
        max_text_len = None
        max_summary_len = None

        # Train file -> self.train_data as tuple (array, max_text_len, max_summary_len)
        if os.path.isfile("cached_data_train.pt"):      # load cached data to avoid encoding again
            train_tuples, max_text_len, max_summary_len = torch.load("cached_data_train.pt")
            print (f"loaded cached train data, "
                   f"len(train_tuples) = {len(train_tuples)}, "
                   f"max_text_len = {max_text_len}, "
                   f"max_summary_len = {max_summary_len}")
        else:
            train_tuples, max_text_len, max_summary_len = self.load_billsumv3(train_path)
            torch.save((train_tuples, max_text_len, max_summary_len), "cached_data_train.pt")

        print(f"num train pairs (text,summary) = {len(train_tuples)}")
        self.train_data = (train_tuples, max_text_len, max_summary_len)
        del train_tuples  # cleanup memory


        # Validation file -> self.val_data as tuple (array, max_text_len, max_summary_len)
        val_tuples = None
        if os.path.isfile("cached_data_val.pt"):      # load cached data to avoid encoding again
            val_tuples, max_text_len, max_summary_len = torch.load("cached_data_val.pt")
            print(f"loaded cached validation data, "
                  f"len(val_tuples) = {len(val_tuples)}, "
                  f"max_text_len = {max_text_len}, "
                  f"max_summary_len = {max_summary_len}")
        else:
            val_tuples, max_text_len, max_summary_len = self.load_billsumv3(val_path)
            torch.save((val_tuples, max_text_len, max_summary_len), "cached_data_val.pt")

        print(f"num val pairs (text,summary) = {len(val_tuples)}")
        self.val_dataset = DatasetSummarizerBillSumv3(val_tuples,
                                                      max_text_len,
                                                      max_summary_len,
                                                      start_token=start_token,
                                                      end_token=end_token,
                                                      pad_token=pad_token,
                                                      vocab=self.word2idx)
        del val_tuples  # cleanup memory


        # Test file -> test dataset
        test_tuples = None
        if os.path.isfile("cached_data_test.pt"):      # load cached data to avoid encoding again
            test_tuples, max_text_len, max_summary_len = torch.load("cached_data_test.pt")
            print(f"loaded cached test data, "
                  f"len(test_tuples) = {len(test_tuples)}, "
                  f"max_text_len = {max_text_len}, "
                  f"max_summary_len = {max_summary_len}")
        else:
            test_tuples, max_text_len, max_summary_len = self.load_billsumv3(test_path)
            torch.save((test_tuples, max_text_len, max_summary_len), "cached_data_test.pt")

        # test_tuples, max_text_len, max_summary_len = self.load_billsumv3(test_path)
        print(f"num test pairs (text,summary) = {len(test_tuples)}")
        self.test_dataset = DatasetSummarizerBillSumv3(test_tuples,
                                                       max_text_len,
                                                       max_summary_len,
                                                       start_token=start_token,
                                                       end_token=end_token,
                                                       pad_token=pad_token,
                                                       vocab=self.word2idx)
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
        src_key_padding_mask = self.nn_model.create_pad_mask(src, self.tokenizer.pad_token_num)
        tgt_key_padding_mask = self.nn_model.create_pad_mask(tgt_input, self.tokenizer.pad_token_num)

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
            print(f"src  = {' '.join(self.tokenizer.decode_seq(src[0, :].view(-1).tolist()))}")
            print(f"tgt  = {' '.join(self.tokenizer.decode_seq(tgt_expected[0, :].view(-1).tolist()))}")
            print(f"pred = {' '.join(self.tokenizer.decode_seq(predicted_sequence))}")
            print("=================================================================")

        return pred, loss


    def predict(self, input_sequence, max_length=10):
        """ Infer sequence from input_sequence """
        self.nn_model.eval()
        y_input = torch.tensor([[start_token_num]], dtype=torch.long, device=self.device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.nn_model.get_tgt_mask(y_input.size(1)).to(self.device)

            pred = self.nn_model(input_sequence, y_input, tgt_mask)

            next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
            next_item = torch.tensor([[next_item]], device=self.device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == end_token_num:
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

            self.train_dataset = DatasetSummarizerBillSumv3(subsample,
                                                            src_max_len=self.train_data[1],
                                                            tgt_max_len=self.train_data[2],
                                                            start_token=start_token,
                                                            end_token=end_token,
                                                            pad_token=pad_token,
                                                            vocab=self.word2idx)
        return super(SummarizerSetup, self).resampling()

