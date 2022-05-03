from training_setup import *
from pathlib import Path
from model_config import TransformerLanguageModelInfo as tlm_info
from model_config import TransformerLanguageModelConfig as tlm_config
from model_config import TransformerLanguageModelDataConfig as tlm_data

if __name__ == "__main__":
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name']).mkdir(parents=True, exist_ok=True)    
    Path('models/' + tlm_info['name'] + '/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name'] + '/best_val_model_so_far').mkdir(parents=True, exist_ok=True)

    train_params = TrainParams(epochs=5,
                               learning_rate=0.0001,
                               inference_max_len=10,
                               grad_norm_clip=0.0,
                               batch_size=64,
                               )

    TS = TrainingSetup(
                    is_gpu=True,
                    is_resume_mode=False,
                    train_params=train_params,
                    )
    TS.load_data(
        train_path=tlm_data['train_path'],
        test_path=tlm_data['test_path'],
        val_path=tlm_data['val_path'],
        )
    TS.run()
    # TS.plot_losses()