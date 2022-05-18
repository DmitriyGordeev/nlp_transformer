from training_setup import *
from pathlib import Path
from model_config import save_model_config

import pandas as pd


if __name__ == "__main__":

    train_param = pd.read_csv('sh_config/sh_train_param.csv')
    model_conf = pd.read_csv('sh_config/sh_model_config.csv')
    param = train_param.join(model_conf.set_index('ID'), on='ID_MODEL', how='inner')

    for i in param.index.values:
        TransformerLanguageModelConfig = {
            'd_model': int(param['d_model'][i]),
            'nhead': int(param['nhead'][i]),
            'num_encoder_layers': int(param['num_encoder_layers'][i]),
            'num_decoder_layers': int(param['num_decoder_layers'][i]),
            'dim_feedforward': int(param['dim_feedforward'][i]),
            'dropout_p': float(param['dropout_p'][i]),
        }

        TransformerLanguageModelDataConfig = {
            'train_path': param['train_path'][i],
            'test_path': param['test_path'][i],
            'val_path': param['val_path'][i],
            'seq_length': int(param['seq_length'][i]),
        }

        TransformerLanguageModelInfo = {
            'name': 'models/' + 'Model_P' + str(param['ID'][i]) + '_M' + str(param['ID_MODEL'][i]),
        }

        TransformerLanguageModelTrainConfig = {
            'epochs': int(param['epochs'][i]),
            'learning_rate': float(param['learning_rate'][i]),
            'inference_max_len': int(param['inference_max_len'][i]),
            'grad_norm_clip': float(param['grad_norm_clip'][i]),
            'batch_size': int(param['batch_size'][i]),
            'weight_decay': float(param['weight_decay'][i])
            # L2 norm coeff: the bigger -> the less overfitting but the slower training
        }

        dict_param = {'TransformerLanguageModelConfig': TransformerLanguageModelConfig,
                      'TransformerLanguageModelDataConfig': TransformerLanguageModelDataConfig,
                      'TransformerLanguageModelInfo': TransformerLanguageModelInfo,
                      'TransformerLanguageModelTrainConfig': TransformerLanguageModelTrainConfig,
                      }


        Path('models').mkdir(parents=True, exist_ok=True)
        Path('models/' + TransformerLanguageModelInfo['name']).mkdir(parents=True, exist_ok=True)
        Path('models/' + TransformerLanguageModelInfo['name'] + '/checkpoints').mkdir(parents=True, exist_ok=True)
        Path('models/' + TransformerLanguageModelInfo['name'] + '/best_val_model_so_far').mkdir(parents=True,
                                                                                                exist_ok=True)

        save_model_config('models/' + TransformerLanguageModelInfo['name'], dict_param)

        model_params = ModelParams(d_model=int(param['d_model'][i]),
                                   nhead=int(param['nhead'][i]),
                                   num_encoder_layers=int(param['num_encoder_layers'][i]),
                                   num_decoder_layers=int(param['num_decoder_layers'][i]),
                                   dim_feedforward=int(param['dim_feedforward'][i]),
                                   dropout_p=float(param['dropout_p'][i]))

        train_params = TrainParams(epochs=int(param['epochs'][i]),
                                   learning_rate=float(param['learning_rate'][i]),
                                   inference_max_len=int(param['inference_max_len'][i]),
                                   grad_norm_clip=float(param['grad_norm_clip'][i]),
                                   batch_size=int(param['batch_size'][i]),
                                   weight_decay=float(param['weight_decay'][i]),
                                   seq_length=int(param['seq_length'][i]),
                                   path_nm=TransformerLanguageModelInfo['name'])

        TS = TrainingSetup(
            is_gpu=True,
            is_resume_mode=False,
            train_params=train_params,
            model_params=model_params,
        )
        TS.load_data(
            train_path=param['train_path'][i],
            test_path=param['test_path'][i],
            val_path=param['val_path'][i],
        )
        TS.run()
        # TS.plot_losses()
