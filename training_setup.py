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
from tokenizer import Tokenizer

matplotlib.use("Agg")


class TrainParams:
    def __init__(self, epochs, learning_rate, inference_max_len, grad_norm_clip, batch_size, val_batch_size):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_inference_len = inference_max_len      # limit on max elements to predict on the final test stage
        self.grad_norm_clip = grad_norm_clip
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size    # if validation set is too big
                                                # we process it by batches to avoid cpu/gpu memory overflow


class TrainingSetup:
    def __init__(self, is_gpu, is_resume_mode, train_params: TrainParams):
        self.device = "cpu"
        if is_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is {self.device}")

        self.is_resume_mode = is_resume_mode

        self.tokenizer = None
        self.vocab = None
        self.vocab_size = 0

        self.train_data = None
        self.val_data = None
        self.test_data = None

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


    def load_data(self, text_filename):
        """ Reads file, tokenize and prepare tensors to train """
        self.tokenizer = Tokenizer()
        with open(text_filename, "r", encoding="utf-8") as f:
            text = f.read()
            sentences, self.vocab = self.tokenizer.assemble_vocab(text)
        self.vocab_size = len(list(self.vocab.keys()))

        # split into train / test by specifying portions e.g. 0.7 - 70% train, the rest 30% - validation:
        train_portion = 0.7
        num_train_samples = int(len(sentences) * train_portion)

        # val and test are divided equally:
        num_val_samples  = int(0.5 * (len(sentences) - num_train_samples))
        num_test_samples = len(sentences) - num_train_samples - num_val_samples

        # Setup tensors
        self.train_data = torch.tensor(sentences[:num_train_samples], dtype=torch.long)
        self.val_data  = torch.tensor(sentences[num_train_samples : num_train_samples + num_val_samples], dtype=torch.long)
        self.test_data = torch.tensor(sentences[num_train_samples + num_val_samples:], dtype=torch.long)

        print (f"")
        print (f"Train samples {num_train_samples}")
        print (f"Validation samples {num_val_samples}")
        print (f"Test samples {num_test_samples}")


    def setup_nn(self):
        self.nn_model = model.Transformer(num_tokens=self.vocab_size,
                                          dim_model=1,
                                          num_heads=1,
                                          num_encoder_layers=1,
                                          num_decoder_layers=1,
                                          dropout_p=0.01).to(self.device)
        self.nn_model.to(self.device)
        print (f"\nParameters in the model = {self.nn_model.count_params()}\n")


    def setup_optimizers(self):
        self.optimizer = top.Adam(self.nn_model.parameters(), lr=self.train_params.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = top.lr_scheduler.StepLR(self.optimizer,
                                                 step_size=100,
                                                 gamma=1.0)


    def train(self):
        start_epoch = 0

        # If we specified resume mode - load checkpoint
        if self.is_resume_mode:
            start_epoch = self.load_checkpoint("checkpoints/checkpoint.pt")
            self.train_params.epochs += start_epoch
            print("resume from epoch:", start_epoch, " till epoch:", self.train_params.epochs)
        else:
            print("training from scratch")

        # split train data into batches
        train_batches = torch.split(self.train_data, self.train_params.batch_size, dim=0)

        for i_epoch in range(start_epoch, self.train_params.epochs):

            epoch_loss = 0      # reset before each new epoch

            for batch_index, batch in enumerate(train_batches):
                src = batch[:, 0, :].to(self.device)
                tgt = batch[:, 1, :].to(self.device)

                self.optimizer.zero_grad()
                pred, loss = self.nn_forward(src, tgt)
                loss.backward()

                # reduce the gradient step if necessary (and too big)
                if self.train_params.grad_norm_clip != 0:
                    self.clip_grad_norm()

                grad_norm = self.get_grad_norm()
                self.optimizer.step()

                # print status
                print(f" epoch {i_epoch} / {self.train_params.epochs - 1},"
                      f" batch {batch_index} / {len(train_batches) - 1},"
                      f" loss = {loss.item()} |"
                      f" gradient norm = {grad_norm}")

                # Accumulate loss obtained for every batch
                with torch.no_grad():
                    epoch_loss += loss.item()

            # get average loss across all batches in this epoch
            epoch_loss = float(epoch_loss) / len(train_batches)
            self.recorded_train_loss.append(epoch_loss)
            if len(self.recorded_train_loss) > 2000:
                self.recorded_train_loss = self.recorded_train_loss[-2000:]

            print("Train loss ", epoch_loss, "....")

            self.validate(i_epoch)

            self.scheduler.step()

            # Saving training snapshot every 20 epochs
            # snapshot = (epoch + model's params + optimizer + scheduler)
            if i_epoch % 20 == 0:
                self.save_checkpoint(i_epoch, "checkpoints/checkpoint.pt")      # this overwrites previously saved snapshot
                self.remove_checkpoint_log_files("checkpoints/*.log")
                f_write_epoch = open("checkpoints/" + str(i_epoch) + ".log",
                                     "w")  # just create a file to indicate what epoch number was saved the last
                f_write_epoch.close()


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


    def validate(self, i_epoch):
        with torch.no_grad():

            batches = torch.split(self.val_data, self.train_params.val_batch_size, dim=0)
            validation_loss = 0

            for idx, batch in enumerate(batches):
                src = batch[:, 0, :].to(self.device)
                tgt = batch[:, 1, :].to(self.device)

                pred, loss = self.nn_forward(src, tgt)

                print("\t val batch =", idx, "/", len(batches), "| loss =", loss.item())
                validation_loss += loss.item()

            validation_loss = float(validation_loss) / len(batches)

            self.recorded_val_loss.append(validation_loss)
            if len(self.recorded_val_loss) > 2000:
                self.recorded_val_loss = self.recorded_val_loss[-2000:]

            print(f"Mean val loss {validation_loss}")
            print("................................................")

            # save the best so far validation loss checkpoint:
            if validation_loss < self.best_val_loss_so_far or self.best_val_loss_so_far == -1:
                print (f"[Saving best validation checkpoint] "
                       f"previous val loss = {self.best_val_loss_so_far},"
                       f" current = {validation_loss}")
                self.save_checkpoint(i_epoch, f"best_val_model_so_far/checkpoint.{i_epoch}.pt")
                self.best_val_loss_so_far = validation_loss


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
        self.test()


    def set_optimizer_lr(self, new_learning_rate):
        """ Alters optimizer's learning rate on the go if necessary """
        for g in self.optimizer.param_groups:
            g['lr'] = new_learning_rate

