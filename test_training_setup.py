import unittest
from training_setup import *


class TestTrainingSetup(unittest.TestCase):

    def test_run_training_setup(self):
        train_params = TrainParams(epochs=10,
                                   learning_rate=0.01,
                                   batch_size=32,
                                   val_batch_size=32,
                                   grad_norm_clip=2.0)

        TS = TrainingSetup(is_gpu=True, is_resume_mode=False, train_params=train_params)
        TS.load_data("saturn.txt")
        # TS.run()

