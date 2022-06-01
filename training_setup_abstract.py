import torch
import torch.nn as nn
import torch.optim as top
# from model_config import TransformerLanguageModelInfo as tlm_info
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plot
import model

class ModelParams:
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout_p: float
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout_p


class TrainParams:
    def __init__(
            self,
            epochs: int,
            learning_rate: float,
            inference_max_len: int,
            grad_norm_clip: float,
            batch_size: int,
            weight_decay: float,
            seq_length: int,
            path_nm: str,

    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.inference_max_len = inference_max_len
        self.grad_norm_clip = grad_norm_clip
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seq_length = seq_length
        self.path_nm = path_nm


class TrainingSetup:
    def __init__(
            self,
            is_gpu: bool,
            is_resume_mode: bool,
            train_params: TrainParams,
            model_params: ModelParams,
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

        self.pretrained_embedding = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.train_params = train_params
        self.model_params = model_params
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

        self.prev_val_loss = 0
        self.best_val_increase_counter = 0      # Counts up every time val loss is greater than the previous
        self.best_val_counter_limit = 5         # if counter greater than this limit we save the checkpoint on validation
        self.val_on_save = -1                   # Validation loss at the moment we saved the checkpoint


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
    ):
        return

    def setup_nn(self):
        self.nn_model = model.TransformerLanguageModel(
            num_tokens=self.word2idx_size,
            d_model=self.model_params.d_model,
            nhead=self.model_params.nhead,
            num_encoder_layers=self.model_params.num_encoder_layers,
            num_decoder_layers=self.model_params.num_decoder_layers,
            dim_feedforward=self.model_params.dim_feedforward,
            dropout_p=self.model_params.dropout_p,
        ).to(self.device)
        self.nn_model.to(self.device)
        print (f"\nParameters in the model = {self.nn_model.count_params()}\n")


    def setup_optimizers(self):
        # self.optimizer = top.RMSprop(self.nn_model.parameters(), lr=self.train_params.learning_rate)
        self.optimizer = top.Adam(self.nn_model.parameters(),
                                  lr=self.train_params.learning_rate,
                                  weight_decay=self.train_params.weight_decay)

        self.criterion = nn.CrossEntropyLoss()

        # self.scheduler = top.lr_scheduler.StepLR(self.optimizer,
        #                                          step_size=50,
        #                                          gamma=0.95)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    patience=5,
                                                                    threshold=0.001,
                                                                    factor=0.5)

    def nn_forward(self, batch, print_enabled=False):
        return


    def train(self):

        dashboard = SummaryWriter()

        start_epoch = 0

        # If we specified resume mode - load checkpoint
        if self.is_resume_mode:
            start_epoch = self.load_checkpoint('models/' + self.train_params.path_nm + '/checkpoints/')
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
            print(f'\n------- epoch {i_epoch} / {self.train_params.epochs - 1} -------')

            for batch_idx, batch in enumerate(dataloader_train):

                if batch_idx % 1 == 0:
                    print (f"\r\t(train) batch = {batch_idx} / {len(dataloader_train)}", end='')

                __, loss = self.nn_forward(batch)

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

            print('\nTrain loss ', train_loss)

            val_loss = self.validate(i_epoch)

            dashboard.add_scalars('loss', {'train': self.recorded_train_loss[-1], 'val': self.recorded_val_loss[-1]}, i_epoch)

            self.scheduler.step(val_loss)

            print(f"learning_rate = {self.optimizer.param_groups[0]['lr']}")

            # Saving training snapshot every 20 epochs
            # snapshot = (epoch + model's params + optimizer + scheduler)
            if i_epoch % 5 == 0:
                start_time = time.time()
                self.save_checkpoint(i_epoch, 'models/' + self.train_params.path_nm + '/checkpoints/')
                print (f"Saving time = {time.time() - start_time} sec")

        dashboard.close()


    def validate(self, i_epoch):
        dataloader_val = DataLoader(
                                    dataset=self.val_dataset,
                                    batch_size=self.train_params.batch_size,
                                    shuffle=True,
                                    )
        with torch.no_grad():
            val_loss = 0
            for batch_index, batch in enumerate(dataloader_val):
                __, loss = self.nn_forward(batch, print_enabled=False)
                val_loss += loss.item()

                if batch_index % 1 == 0:
                    print (f"\r\t(val) batch = {batch_index} / {len(dataloader_val)}", end='')

            val_loss = float(val_loss) / len(dataloader_val)
            self.recorded_val_loss.append(val_loss)
            print(f'\nValidation loss', val_loss)

            # # save the best so far validation loss checkpoint:
            # if val_loss < self.best_val_loss_so_far or self.best_val_loss_so_far == -1:
            #     time.time()
            #     self.save_checkpoint(i_epoch, 'models/' + tlm_info['name'] + '/best_val_model_so_far/')
            #     self.best_val_loss_so_far = val_loss


            if self.val_on_save > 0 and val_loss < self.val_on_save:
                print (f"[debug] Reaching another loss decreasing region, reset self.val_on_save")
                self.val_on_save = -1

            # saves checkpoint when we're observing val loss on plateau or rising for several times
            if self.val_on_save < 0:
                if self.prev_val_loss != 0 and abs(val_loss - self.prev_val_loss) / self.prev_val_loss <= 0.005:
                    self.best_val_increase_counter += 1
                    print (f"[debug] self.best_val_increase_counter = {self.best_val_increase_counter}")
                    if self.best_val_increase_counter == self.best_val_counter_limit:
                        t = time.time()
                        self.save_checkpoint(i_epoch, 'models/' + self.train_params.path_nm + '/best_val_model_so_far/')
                        self.val_on_save = val_loss
                        self.best_val_increase_counter = 0
                        print (f"Checkpoint saved in {time.time() - t} sec")
                else:
                    # reset accumulated counter
                    self.best_val_increase_counter = 0

            self.prev_val_loss = val_loss

        return val_loss


    def test(self):
        with torch.no_grad():
            dataloader_test = DataLoader(
                                        dataset=self.test_dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        )
            test_loss = 0
            self.nn_model.eval()

            for batch_idx, batch in enumerate(dataloader_test):
                __, loss = self.nn_forward(batch, print_enabled=False)
                test_loss += loss.item()

            test_loss = float(test_loss) / len(dataloader_test)

            self.recorded_test_loss.append(test_loss)
            print(f'\n------- Test loss : {self.recorded_test_loss[-1]} -------')


    def predict(self):
        return


    def save_checkpoint(self, current_epoch: int, directory: str):
        """ Saves model, optimizer and scheduler state in specified dir """
        self.remove_prev_checkpoints(directory)
        checkpoint_state_dict = {
            'epoch': current_epoch,
            'model': self.nn_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(checkpoint_state_dict, directory + f"/checkpoint.{current_epoch}.pt")     # saving epoch, model, optimizer, scheduler (as dicts)
        torch.save(self.nn_model, directory + f"/model.{current_epoch}.pth")    # saving model file separately (don't need to parse dict on load)


    @staticmethod
    def remove_prev_checkpoints(directory: str):
        """ Removes model and checkpoint files in specified dir """
        files = glob.glob(directory + "/*")
        for filePath in files:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)


    def load_checkpoint(self, directory: str):
        """ Searches for checkpoint*.pt and model*.pth files and loads them """
        checkpoint_files = glob.glob(directory + "/checkpoint*")
        if len(checkpoint_files) == 0:
            print("Couldn't find checkpoint* files in: " + directory)
            exit(1)

        model_files = glob.glob(directory + "/model*")
        if len(model_files) == 0:
            print("Couldn't find model* files in: " + directory)
            exit(1)

        print (f"Checkpoint file: {checkpoint_files[0]}")
        print(f"Model file: {model_files[0]}")

        # self.nn_model = torch.load(model_files[0])
        checkpoint_state_dict = torch.load(checkpoint_files[0])
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
