from training_setup import *
from pathlib import Path
from model_config import TransformerLanguageModelInfo as tlm_info
from model_config import TransformerLanguageModelConfig as tlm_config
from model_config import TransformerLanguageModelDataConfig as tlm_data
from model_config import TransformerLanguageModelTrainConfig as tlm_train
from model_config import save_model_config

if __name__ == "__main__":
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name']).mkdir(parents=True, exist_ok=True)    
    Path('models/' + tlm_info['name'] + '/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name'] + '/best_val_model_so_far').mkdir(parents=True, exist_ok=True)

    save_model_config('models/' + tlm_info['name'])

    train_params = TrainParams(epochs=tlm_train['epochs'],
                               learning_rate=tlm_train['learning_rate'],
                               inference_max_len=tlm_train['inference_max_len'],
                               grad_norm_clip=tlm_train['grad_norm_clip'],
                               batch_size=tlm_train['batch_size'],
                               weight_decay=tlm_train['weight_decay'])

    TS = TrainingSetup(
                    is_gpu=True,
                    is_resume_mode=True,
                    train_params=train_params,
                    )
    TS.load_data(
        train_path=tlm_data['train_path'],
        test_path=tlm_data['test_path'],
        val_path=tlm_data['val_path'],
        )
    TS.run()
    # TS.plot_losses()