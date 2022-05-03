from asyncio import constants
import torch
import torch.nn as nn
import torch.optim as top
import glob
import os
import matplotlib
import matplotlib.pyplot as plot
import numpy
import random
import math
import model
from tokenizer import TokenizerLanguageModel, TokenizerCollection
from model_config import TransformerLanguageModelConfig as tlm_conf
from model_config import TransformerLanguageModelDataConfig as tlm_data
from model_config import TransformerLanguageModelInfo as tlm_info
import model_constants
from torch.utils.data import DataLoader
from data_loader import DatasetLanguageModel
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")


class TrainParams:
    def __init__(
            self,
            epochs: int,
            learning_rate: float,
            inference_max_len: int,
            grad_norm_clip: float,
            batch_size: int,
    ):
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.inference_max_len = inference_max_len
        self.grad_norm_clip = grad_norm_clip
        self.batch_size = batch_size


class TrainingSetup:
    def __init__(
            self,
            is_gpu: bool,
            is_resume_mode: bool,
            train_params: TrainParams,
    ):
        
        self.device = "cpu"
        if is_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is {self.device}")

        self.is_resume_mode = is_resume_mode

        self.tokenizer = None
        self.word2idx = None
        self.idx2word = None
        self.word2idx_size = 0

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.train_params = train_params
        self.num_train_size = 0
        self.num_val_size = 0
        self.num_test_size = 0

        self.nn_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.recorded_train_loss = []
        self.recorded_val_loss = []
        self.recorded_test_loss = []
        self.best_val_loss_so_far = -1


    def clip_grad_norm(self):
        """ Clips gradient vector if too high """
        if self.train_params.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), self.train_params.grad_norm_clip)


    def get_grad_norm(self):
        """ Calculates current magnitude of the gradient vector """
        with torch.no_grad():
            gradient_norm = 0
            for p in self.nn_model.parameters():
                if p is not None and p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    gradient_norm += param_norm.item() ** 2
            gradient_norm = gradient_norm ** 0.5
            return gradient_norm


    def load_data(
        self,
        train_path: str,
        test_path: str,
        val_path: str,
    ):
        """ Reads file, tokenize and prepare tensors to train """
        self.tokenizer = TokenizerLanguageModel(
                                            pad_token=model_constants.pad_token,
                                            start_token=model_constants.start_token,
                                            end_token=model_constants.end_token,
                                            unk_token=model_constants.unk_token,
                                            pad_token_num=model_constants.pad_token_num,
                                            start_token_num=model_constants.start_token_num,
                                            end_token_num=model_constants.end_token_num,
                                            unk_token_num=model_constants.unk_token_num,
                                            )
        
        f = open(train_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        self.tokenizer.assemble_vocab(text)
        self.word2idx = self.tokenizer.word2idx
        self.idx2word = self.tokenizer.idx2word
        self.word2idx_size = self.tokenizer.word2idx_size
        self.train_data = self.tokenizer.encode_seq(text)
        f.close()

        f = open(test_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        self.test_data = self.tokenizer.encode_seq(text)
        f.close()

        f = open(val_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        self.val_data = self.tokenizer.encode_seq(text)
        f.close()

        self.train_dataset = DatasetLanguageModel(
                                            data = self.train_data,
                                            sequence_length=tlm_data['seq_length'],
                                            start_token=model_constants.start_token,
                                            end_token=model_constants.end_token,
                                            pad_token=model_constants.pad_token,
                                            vocab=self.word2idx,
                                            )
        self.test_dataset = DatasetLanguageModel(
                                            data = self.test_data,
                                            sequence_length=tlm_data['seq_length'],
                                            start_token=model_constants.start_token,
                                            end_token=model_constants.end_token,
                                            pad_token=model_constants.pad_token,
                                            vocab=self.word2idx,
                                            )
        self.val_dataset = DatasetLanguageModel(
                                            data = self.val_data,
                                            sequence_length=tlm_data['seq_length'],
                                            start_token=model_constants.start_token,
                                            end_token=model_constants.end_token,
                                            pad_token=model_constants.pad_token,
                                            vocab=self.word2idx,
                                            )
        

    def setup_nn(self):
        self.nn_model = model.TransformerLanguageModel(
                                                    num_tokens=self.word2idx_size,
                                                    d_model=tlm_conf['d_model'],
                                                    nhead=tlm_conf['nhead'],
                                                    num_encoder_layers=tlm_conf['num_encoder_layers'],
                                                    num_decoder_layers=tlm_conf['num_decoder_layers'],
                                                    dim_feedforward=tlm_conf['dim_feedforward'],
                                                    dropout_p=tlm_conf['dropout_p'],
                                                    ).to(self.device)
        self.nn_model.to(self.device)
        print (f"\nParameters in the model = {self.nn_model.count_params()}\n")


    def setup_optimizers(self):
        self.optimizer = top.RMSprop(self.nn_model.parameters(), lr=self.train_params.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = top.lr_scheduler.StepLR(self.optimizer,
                                                 step_size=100,
                                                 gamma=1.0)


    def train(self):

        dashboard = SummaryWriter()

        start_epoch = 0

        # If we specified resume mode - load checkpoint
        if self.is_resume_mode:
            start_epoch = self.load_checkpoint('models/' + tlm_info['name'] + '/checkpoints/checkpoint.pt')
            self.train_params.epochs += start_epoch
            print("resume from epoch:", start_epoch, " till epoch:", self.train_params.epochs)
        else:
            print("training from scratch")

        dataloader_train = DataLoader(
                                    dataset=self.train_dataset,
                                    batch_size=self.train_params.batch_size,
                                    shuffle=True,
                                    )
                
        for i_epoch in range(start_epoch, self.train_params.epochs):

            train_loss = 0      # reset before each new epoch

            for batch in dataloader_train:

                src = batch.to(self.device)
                tgt_input = batch[:, :-1].to(self.device)
                tgt_expected = batch[:, 1:].to(self.device)

                # Get mask to mask out the next words
                sequence_length = tgt_input.size(1)
                tgt_mask = self.nn_model.get_tgt_mask(sequence_length).to(self.device)

                # Standard training except we pass in y_input and tgt_mask
                pred = self.nn_model(src, tgt_input, tgt_mask)
        
                # Permute pred to have batch size first again
                pred = pred.permute(0, 2, 1)
                tgt_expected = tgt_expected.type(torch.int64)
        
                loss = self.criterion(pred, tgt_expected)

                self.optimizer.zero_grad()
                loss.backward()

                # reduce the gradient step if necessary (and too big)
                if self.train_params.grad_norm_clip != 0:
                    self.clip_grad_norm()
                
                self.optimizer.step()

                with torch.no_grad():
                    train_loss += loss.item()
            
            train_loss = float(train_loss) / len(dataloader_train)
            
            self.recorded_train_loss.append(train_loss)
            
            dashboard.add_scalar('train loss', train_loss, i_epoch)

            print(f'\n------- epoch {i_epoch} -------')
            print('Train loss ', train_loss)

            self.validate(dashboard, i_epoch)

            self.scheduler.step()

            # Saving training snapshot every 20 epochs
            # snapshot = (epoch + model's params + optimizer + scheduler)
            if i_epoch % 20 == 0:
                self.save_checkpoint(i_epoch, 'models/' + tlm_info['name'] + '/checkpoints/checkpoint.pt')
                self.remove_checkpoint_log_files('models/' + tlm_info['name'] + '/checkpoints/*.log')
                f_write_epoch = open('models/' + tlm_info['name'] + '/checkpoints/' + str(i_epoch) + '.log', 'w')
                f_write_epoch.close()
        
        dashboard.close()


    def nn_forward(self, src, tgt):
        """ Helper function to be invoked everywhere on training, validation and test stages
        :param src:
        :param tgt:
        :return: loss
        """
        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = tgt[:, :-1]
        y_expected = tgt[:, 1:]

        # Get mask to mask out the next words.
        sequence_length = y_input.size(1)
        tgt_mask = self.nn_model.get_tgt_mask(sequence_length).to(self.device)

        # Standard training except we pass in y_input and tgt_mask
        pred = self.nn_model(src, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = self.criterion(pred, y_expected)
        return pred, loss


    def validate(self, dashboard, i_epoch):

        dataloader_val = DataLoader(
                                    dataset=self.val_dataset,
                                    batch_size=self.train_params.batch_size,
                                    shuffle=True,
                                    )
        
        with torch.no_grad():

            val_loss = 0

            for batch in dataloader_val:

                src = batch.to(self.device)
                tgt_input = batch[:, :-1].to(self.device)
                tgt_expected = batch[:, 1:].to(self.device)

                # Get mask to mask out the next words
                sequence_length = tgt_input.size(1)
                tgt_mask = self.nn_model.get_tgt_mask(sequence_length).to(self.device)

                # Standard training except we pass in y_input and tgt_mask
                pred = self.nn_model(src, tgt_input, tgt_mask)
        
                # Permute pred to have batch size first again
                pred = pred.permute(0, 2, 1)
                tgt_expected = tgt_expected.type(torch.int64)
        
                loss = self.criterion(pred, tgt_expected)

                val_loss += loss.item()

            val_loss = float(val_loss) / len(dataloader_val)

            self.recorded_val_loss.append(val_loss)

            dashboard.add_scalar('val loss', val_loss, i_epoch)

            print('Validation loss ', val_loss)
            
            # save the best so far validation loss checkpoint:
            if val_loss < self.best_val_loss_so_far or self.best_val_loss_so_far == -1:
                self.save_checkpoint(i_epoch, 'models/' + tlm_info['name'] + f'/best_val_model_so_far/best_checkpoint.pt')
                self.best_val_loss_so_far = val_loss


    def test(self):
        

        test_loss = 0
        batches = torch.split(self.test_data, 1, dim=0)
        self.nn_model.eval()

        for idx, batch in enumerate(batches):
            src = batch[:, 0, :].to(self.device)
            tgt = batch[:, 1, :].to(self.device)

            # calculate loss
            pred_matrix, loss = self.nn_forward(src, tgt)

            # predict completely unseen sequence
            predicted_sequence = self.predict(src, max_length=self.train_params.max_inference_len)

            # decode src, tgt and prediction into human-readable string
            print (f"Test sample idx {idx}, max_inference_len = {self.train_params.max_inference_len} : ")
            print (f"src  = {self.tokenizer.decode_sentence(src[0, :].view(-1).tolist())}")
            print (f"tgt  = {self.tokenizer.decode_sentence(tgt[0, :].view(-1).tolist())}")
            print (f"pred = {self.tokenizer.decode_sentence(predicted_sequence)}\n")

            print("Test batch ", idx, "/", len(batches) - 1, ", loss =", loss.item())
            test_loss += loss.item()

        print("................................................")
        print(f"mean test loss = {float(test_loss) / len(batches)}")


    def predict(self, input_sequence, max_length=10):
        """ Infer sequence from input_sequence """
        self.nn_model.eval()
        y_input = torch.tensor([[self.tokenizer.SOS]], dtype=torch.long, device=self.device)

        for _ in range(max_length):
            # Get source mask
            tgt_mask = self.nn_model.get_tgt_mask(y_input.size(1)).to(self.device)

            pred = self.nn_model(input_sequence, y_input, tgt_mask)

            # next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = pred[-1, :, :].argmax().item()  # num of the highest proba on the last dimension
            next_item = torch.tensor([[next_item]], device=self.device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)

            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == self.tokenizer.EOS:
                break

        return y_input.view(-1).tolist()


    def save_checkpoint(self, current_epoch, checkpoint_filepath):
        checkpoint_state_dict = {
            'epoch': current_epoch,
            'model': self.nn_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        checkpoint_filepath = checkpoint_filepath
        torch.save(checkpoint_state_dict, checkpoint_filepath)

    @staticmethod
    def remove_checkpoint_log_files(pattern):
        files = glob.glob(pattern)
        for filePath in files:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)


    def load_checkpoint(self, checkpoint_fpath):
        if not os.path.isfile(checkpoint_fpath):
            print (f"Checkpoint file {checkpoint_fpath} not found")
            exit(1)
        checkpoint_state_dict = torch.load(checkpoint_fpath)
        self.nn_model.load_state_dict(checkpoint_state_dict['model'])
        self.optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_state_dict['scheduler'])
        return checkpoint_state_dict['epoch']


    def plot_losses(self, last_n=-1):
        tl = self.recorded_train_loss
        vl = self.recorded_val_loss
        if last_n > 0:
            tl = tl[-last_n:]
            vl = vl[-last_n:]

        plot.figure(figsize=(12, 8), dpi=100)

        plot.plot(tl, 'g.')
        plot.plot(tl, 'g', label="train")

        plot.plot(vl, 'b.')
        plot.plot(vl, 'b', label="val")
        plot.xlabel("epoch")
        plot.ylabel("loss")
        plot.grid()
        plot.legend()

        if last_n == -1:
            plot.savefig("loss.jpg")
        else:
            plot.savefig(f"loss.{last_n}.jpg")

        plot.close()
        plot.clf()


    def run(self):
        """ encapsulates other functions and runs them in the right order """
        self.setup_nn()
        self.setup_optimizers()
        self.train()
        # self.test()


    def set_optimizer_lr(self, new_learning_rate):
        """ Alters optimizer's learning rate on the go if necessary """
        for g in self.optimizer.param_groups:
            g['lr'] = new_learning_rate

