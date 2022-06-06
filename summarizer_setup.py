import torch
import glob
import matplotlib
import numpy as np
import random

from transformer_utils import training_setup_abstract
from transformer_utils import model
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection

# TODO: change usage of ModelConfig
from summarizer_model_config import TransformerSummarizerModelConfig as tlm_conf
from summarizer_model_config import TransformerSummarizerModelDataConfig as tlm_data
from summarizer_model_config import TransformerSummarizerModelInfo as tlm_info

from transformer_utils.model_constants import *
from transformer_utils.data_loader import DatasetSummarizerModel

matplotlib.use("Agg")


class ModelParams(training_setup_abstract.ModelParams):
    pass


class TrainParams(training_setup_abstract.TrainParams):
    pass


class TrainingSetup(training_setup_abstract.TrainingSetup):

    @staticmethod
    def read_files(src_dir: str, tgt_dir: str):

        # src_path_list = glob.glob('data/src/*.txt')
        src_path_list = glob.glob(src_dir + "/*.txt")
        src_list = []
        for el in src_path_list:
            f = open(el, 'r', encoding='utf-8')
            text = f.read()
            src_list.append(text)
            f.close()

        # tgt_path_list = glob.glob('data/tgt/*.txt')
        tgt_path_list = glob.glob(tgt_dir + '/*.txt')
        tgt_list = []
        for el in tgt_path_list:
            f = open(el, 'r', encoding='utf-8')
            text = f.read()
            tgt_list.append(text)
            f.close()

        if len(src_list) != len(tgt_list):
            raise RuntimeError('the number of src samples is not equal to the number of tgt samples')

        return src_list, tgt_list

    @staticmethod
    def train_val_test_split(
            src: list,
            tgt: list,
            ratio=(7, 2, 1),
    ):
        seq_len = len(src)
        weight = []
        s = 0
        for el in (7, 2, 1):
            if el < 0:
                raise ValueError('ratio elements must be >= 0')
            s += el
            weight.append(s)
        weight = [int(w * seq_len / s) for w in weight]

        idx = np.arange(seq_len)
        random.shuffle(idx)

        train_idx = idx[:weight[0]]
        val_idx = idx[weight[0]:weight[1]]
        test_idx = idx[weight[1]:]

        train_src = [src[i] for i in train_idx]
        val_src = [src[i] for i in val_idx]
        test_src = [src[i] for i in test_idx]

        train_tgt = [tgt[i] for i in train_idx]
        val_tgt = [tgt[i] for i in val_idx]
        test_tgt = [tgt[i] for i in test_idx]

        src_out = [train_src, val_src, test_src]
        tgt_out = [train_tgt, val_tgt, test_tgt]

        return src_out, tgt_out

    def data_cleanup(
            self,
            src,
            tgt,
            tokenizer,
    ):
        src_out = []
        tgt_out = []

        src_max_len = -1
        tgt_max_len = -1

        for i in range(len(src)):
            src_out_i = []
            tgt_out_i = []

            for j in range(len(src[i])):
                src_out_i.append(self.tokenizer.cleanup(src[i][j], tokenizer))
                tgt_out_i.append(self.tokenizer.cleanup(tgt[i][j], tokenizer))

                src_max_len = max(src_max_len, len(src_out_i[-1]))
                tgt_max_len = max(tgt_max_len, len(tgt_out_i[-1]))

            src_out.append(src_out_i)
            tgt_out.append(tgt_out_i)

        words = []
        for i in range(len(src_out[0])):
            for el in src_out[0][i]:
                words.append(el)

        for i in range(len(tgt_out[0])):
            for el in tgt_out[0][i]:
                words.append(el)

        return src_out, tgt_out, words, src_max_len, tgt_max_len

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

        src_text, tgt_text = self.read_files()
        src_sample, tgt_sample = self.train_val_test_split(src_text, tgt_text)
        src_seq, tgt_seq, words, self.src_max_len, self.tgt_max_len = self.data_cleanup(src_sample, tgt_sample,
                                                                                        TokenizerCollection.basic_english_by_word)

        self.tokenizer.assemble_vocab(words)
        self.word2idx = self.tokenizer.word2idx
        self.idx2word = self.tokenizer.idx2word
        self.word2idx_size = self.tokenizer.word2idx_size

        torch.save(self.word2idx, 'models/' + tlm_info['name'] + '/vocab.pt')

        self.train_data = [[self.tokenizer.encode_seq(seq) for seq in src_seq[0]],
                           [self.tokenizer.encode_seq(seq) for seq in tgt_seq[0]]]
        self.val_data = [[self.tokenizer.encode_seq(seq) for seq in src_seq[1]],
                         [self.tokenizer.encode_seq(seq) for seq in tgt_seq[1]]]
        self.test_data = [[self.tokenizer.encode_seq(seq) for seq in src_seq[2]],
                          [self.tokenizer.encode_seq(seq) for seq in tgt_seq[2]]]

        self.train_dataset = DatasetSummarizerModel(
            data=self.train_data,
            src_max_len=self.src_max_len,
            tgt_max_len=self.tgt_max_len,
            start_token=special_tokens['tokens']['start_token'],
            end_token=special_tokens['tokens']['end_token'],
            pad_token=special_tokens['tokens']['pad_token'],
            vocab=self.word2idx,
        )

        self.val_dataset = DatasetSummarizerModel(
            data=self.val_data,
            src_max_len=self.src_max_len,
            tgt_max_len=self.tgt_max_len,
            start_token=special_tokens['tokens']['start_token'],
            end_token=special_tokens['tokens']['end_token'],
            pad_token=special_tokens['tokens']['pad_token'],
            vocab=self.word2idx,
        )

        self.test_dataset = DatasetSummarizerModel(
            data=self.test_data,
            src_max_len=self.src_max_len,
            tgt_max_len=self.tgt_max_len,
            start_token=special_tokens['tokens']['start_token'],
            end_token=special_tokens['tokens']['end_token'],
            pad_token=special_tokens['tokens']['pad_token'],
            vocab=self.word2idx,
        )


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
        y_input = torch.tensor([[special_tokens.start_token_num]], dtype=torch.long, device=self.device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.nn_model.get_tgt_mask(y_input.size(1)).to(self.device)

            pred = self.nn_model(input_sequence, y_input, tgt_mask)

            next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
            next_item = torch.tensor([[next_item]], device=self.device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == special_tokens.end_token_num:
                break

        return y_input.view(-1).tolist()

