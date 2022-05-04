from training_setup import *
from pathlib import Path
from model_config import TransformerLanguageModelInfo as tlm_info
from model_config import TransformerLanguageModelConfig as tlm_config
from model_config import TransformerLanguageModelDataConfig as tlm_data
from model_config import TransformerLanguageModelTrainConfig as tlm_train

if __name__ == "__main__":
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name']).mkdir(parents=True, exist_ok=True)    
    Path('models/' + tlm_info['name'] + '/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name'] + '/best_val_model_so_far').mkdir(parents=True, exist_ok=True)

    train_params = TrainParams(epochs=tlm_train['epochs'],
                               learning_rate=tlm_train['learning_rate'],
                               inference_max_len=tlm_train['inference_max_len'],
                               grad_norm_clip=tlm_train['grad_norm_clip'],
                               batch_size=tlm_train['batch_size'],
                               )

    TS = TrainingSetup(
                    is_gpu=False,
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