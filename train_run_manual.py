# from classifier_setup import *
# from langmodel_setup import *
from summarizer_setup import *
from pathlib import Path

# TODO: make single model config file for all models (?)
from summarizer_model_config import TransformerSummarizerModelInfo as tlm_info
from summarizer_model_config import TransformerSummarizerModelConfig as tlm_conf
from summarizer_model_config import TransformerSummarizerModelDataConfig as tlm_data
from summarizer_model_config import TransformerSummarizerModelTrainConfig as tlm_train
from summarizer_model_config import save_model_config

# from summarizer_model_config import *

if __name__ == "__main__":
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name']).mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name'] + '/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('models/' + tlm_info['name'] + '/best_val_model_so_far').mkdir(parents=True, exist_ok=True)

    save_model_config('models/' + tlm_info['name'])

    model_params = ModelParams(d_model=tlm_conf['d_model'],
                               nhead=tlm_conf['nhead'],
                               num_encoder_layers=tlm_conf['num_encoder_layers'],
                               num_decoder_layers=tlm_conf['num_decoder_layers'],
                               dim_feedforward=tlm_conf['dim_feedforward'],
                               dropout_p=tlm_conf['dropout_p'])

    train_params = TrainParams(epochs=tlm_train['epochs'],
                               learning_rate=tlm_train['learning_rate'],
                               inference_max_len=tlm_train['inference_max_len'],
                               grad_norm_clip=tlm_train['grad_norm_clip'],
                               batch_size=tlm_train['batch_size'],
                               weight_decay=tlm_train['weight_decay'],
                               seq_length=-1,
                               path_nm=tlm_info['name'])

    TS = SummarizerSetup(is_gpu=True,
                         is_resume_mode=False,
                         train_params=train_params,
                         model_params=model_params)

    TS.load_data(
        train_path="data/summarizer_billsum_v3/us_train_data_final_OFFICIAL.jsonl",
        val_path="data/summarizer_billsum_v3/us_test_data_final_OFFICIAL.jsonl",
        test_path="data/summarizer_billsum_v3/ca_test_data_final_OFFICIAL.jsonl",
    )

    TS.run()



    # TS = ClassiferSetup(is_gpu=True,
    #                     is_resume_mode=False,
    #                     train_params=train_params,
    #                     model_params=model_params)
    #
    # TS.load_data(
    #     train_path="data/classification/train.csv",
    #     val_path="data/classification/val.csv",
    #     test_path="data/classification/test.csv",
    # )
    #
    # TS.run()

    # TS = LangModelSetup(
    #     is_gpu=True,
    #     is_resume_mode=False,
    #     train_params=train_params,
    #     model_params=model_params,
    # )
    #
    # TS.load_data(
    #     train_path=tlm_data['train_path'],
    #     test_path=tlm_data['test_path'],
    #     val_path=tlm_data['val_path'],
    # )
    #
    # TS.run()

    # TS = TrainingSetup(
    #     is_gpu=True,
    #     is_resume_mode=True,
    #     train_params=train_params,
    #     model_params=model_params,
    # )
    # TS.load_data(
    #     train_path=tlm_data['train_path'],
    #     test_path=tlm_data['test_path'],
    #     val_path=tlm_data['val_path'],
    # )
    # TS.run()
    # # TS.plot_losses()
