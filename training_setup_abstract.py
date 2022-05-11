import torch
from model_config import TransformerLanguageModelInfo as tlm_info
from torch.utils.tensorboard import SummaryWriter

class TrainParams:
    def __init__(
            self,
            epochs: int,
            learning_rate: float,
            grad_norm_clip: float,
            batch_size: int,
            inference_max_len=None,
    ):        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grad_norm_clip = grad_norm_clip
        self.batch_size = batch_size
        self.inference_max_len = inference_max_len


class TrainingSetup:
    def __init__(
        self,
        is_gpu: bool,
        is_resume_mode: bool,
        train_params: TrainParams,            
    ):
        # self.tokenizer = None
        # self.word2idx = None
        # self.idx2word = None
        # self.word2idx_size = 0

        self.train_data_iterator = None
        self.val_data_iterator = None
        self.test_data_iterator = None
        
        # self.train_dataset = None
        # self.test_dataset = None
        # self.val_dataset = None

        self.train_params = train_params
        # self.num_train_size = 0
        # self.num_val_size = 0
        # self.num_test_size = 0

        self.nn_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.recorded_train_loss = []
        self.recorded_val_loss = []
        self.recorded_test_loss = []
        self.best_val_loss_so_far = None
        return
        

    def clip_grad_norm(
        self,
    ):
        """ Clips gradient vector if too high """
        if self.train_params.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), self.train_params.grad_norm_clip)
        return


    def get_grad_norm(
        self,
    ):
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
        train_data_iterator,
        val_data_iterator,
        test_data_iterator,        
    ):
        self.train_data_iterator = train_data_iterator
        self.val_data_iterator = val_data_iterator
        self.test_data_iterator = test_data_iterator
        
        return
        

    def setup_nn(
        self,
        model,
    ):
        self.nn_model = model
        self.nn_model.to(self.device)
        print (f"\nparameters in the model: {self.nn_model.count_params()}\n")
        return


    def setup_optimizers(
        self,
        optimizer,
        criterion,
        scheduler=None,
    ):
        self.optimizer = optimizer(self.nn_model.parameters(), lr=self.train_params.learning_rate)
        self.criterion = criterion
        self.scheduler = scheduler
        return


    def nn_forward(
        self,
    ):
        return


    def train(
        self,
    ):
        dashboard = SummaryWriter()

        start_epoch = 0

        # # If we specified resume mode - load checkpoint
        # if self.is_resume_mode:
        #     start_epoch = self.load_checkpoint('models/' + tlm_info['name'] + '/checkpoints/checkpoint.pt')
        #     self.train_params.epochs += start_epoch
        #     print("resume from epoch:", start_epoch, " till epoch:", self.train_params.epochs)
        # else:
        #     print("training from scratch")
                
        for epoch in range(start_epoch, self.train_params.epochs):

            train_loss = 0
            for batch in self.train_data_iterator:

                __, loss = self.nn_forward(batch)

                self.optimizer.zero_grad()
                loss.backward()

                # reduce the gradient step if necessary (and too big)
                if self.train_params.grad_norm_clip != 0:
                    self.clip_grad_norm()
                
                self.optimizer.step()

                with torch.no_grad():
                    train_loss += loss.item()
            
            train_loss = float(train_loss) / len(self.train_data_iterator)
            
            self.recorded_train_loss.append(train_loss)

            print(f'\n------- epoch {epoch} -------')
            print('train loss ', train_loss)

            self.validate(epoch)

            dashboard.add_scalars('loss', {'train': self.recorded_train_loss[-1], 'val': self.recorded_val_loss[-1]}, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # # Saving training snapshot every 20 epochs
            # # snapshot = (epoch + model's params + optimizer + scheduler)
            # if i_epoch % 20 == 0:
            #     self.save_checkpoint(i_epoch, 'models/' + tlm_info['name'] + '/checkpoints/checkpoint.pt')
            #     self.remove_checkpoint_log_files('models/' + tlm_info['name'] + '/checkpoints/*.log')
            #     f_write_epoch = open('models/' + tlm_info['name'] + '/checkpoints/' + str(i_epoch) + '.log', 'w')
            #     f_write_epoch.close()
        
        dashboard.close()
        return
    

    def validate(
        self,
        epoch,
    ):
        with torch.no_grad():

            val_loss = 0

            for batch in self.val_data_iterator:
                
                __, loss = self.nn_forward(batch)

                val_loss += loss.item()

            val_loss = float(val_loss) / len(self.val_data_iterator)

            self.recorded_val_loss.append(val_loss)

            print('val loss ', val_loss)
            
            # save the best so far validation loss checkpoint:
            if val_loss < self.best_val_loss_so_far or self.best_val_loss_so_far is None:
                self.save_checkpoint(epoch, 'models/' + tlm_info['name'] + f'/best_val_model_so_far/best_checkpoint.pt')
                self.best_val_loss_so_far = val_loss
                torch.save(self.nn_model, 'models/' + tlm_info['name'] + '/model.pth')

        return


    def test(
        self,
    ):
        with torch.no_grad():
        
            test_loss = 0
            self.nn_model.eval()

            for batch in self.test_data_iterator:

                __, loss = self.nn_forward(batch)

                test_loss += loss.item()

            test_loss = float(test_loss) / len(self.test_data_iterator)

            self.recorded_test_loss.append(test_loss)
            print('\n------- training is finished -------')
            print(f'test loss : {self.recorded_test_loss[-1]}')
        return    


    def save_checkpoint(
        self,
    ):
        return
    

    @staticmethod
    def remove_checkpoint_log_files():
        return


    def load_checkpoint(
        self,
    ):
        return


    def plot_losses(
        self,
    ):
        return


    def run(
        self,
    ):
        self.setup_nn()
        self.setup_optimizers()
        self.train()
        self.test()
        return


    def set_optimizer_lr(
        self,
        new_learning_rate,
    ):
        """ Alters optimizer's learning rate on the go if necessary """
        for g in self.optimizer.param_groups:
            g['lr'] = new_learning_rate
        return

