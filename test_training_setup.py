import unittest
from training_setup import *


class TestTrainingSetup(unittest.TestCase):

    def test_run_training_setup(self):
        train_params = TrainParams(epochs=100,
                                   learning_rate=0.0001,
                                   batch_size=64,
                                   val_batch_size=64,
                                   grad_norm_clip=0.0,
                                   inference_max_len=10)

        TS = TrainingSetup(is_gpu=True, is_resume_mode=False, train_params=train_params)
        TS.load_data("saturn.txt")
        TS.run()
        TS.plot_losses()

