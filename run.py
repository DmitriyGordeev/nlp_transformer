from training_setup import *
from pathlib import Path

if __name__ == "__main__":
    Path("best_val_model_so_far").mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    train_params = TrainParams(epochs=100,
                               learning_rate=0.0001,
                               batch_size=64,
                               val_batch_size=64,
                               inference_max_len=10,
                               grad_norm_clip=0.0)

    TS = TrainingSetup(is_gpu=True, is_resume_mode=False, train_params=train_params)
    TS.load_data("data/space.txt")
    TS.run()
    TS.plot_losses()