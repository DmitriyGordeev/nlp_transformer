from training_setup import *

if __name__ == "__main__":
    train_params = TrainParams(epochs=120,
                               learning_rate=0.0001,
                               batch_size=64,
                               val_batch_size=64,
                               inference_max_len=10,
                               grad_norm_clip=0.0)

    TS = TrainingSetup(is_gpu=True, is_resume_mode=True, train_params=train_params)
    TS.load_data("saturn.txt")
    TS.run()
    TS.plot_losses()