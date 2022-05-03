TransformerLanguageModelConfig = {
    'd_model': 64,
    'nhead': 16,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 1024,
    'dropout_p': 0.1,
}

TransformerLanguageModelDataConfig = {
    'train_path': 'data/TheWarOfTheWorlds.txt',
    'test_path': 'data/InTheDaysOfTheComet.txt',
    'val_path': 'data/TheTimeMachine.txt',
    'seq_length': 128,
}

TransformerLanguageModelInfo = {
    'name': 'model1',
}

TransformerLanguageModelTrainConfig = {
    'epochs': 5,
    'learning_rate': 0.0001,
    'inference_max_len': 10,
    'grad_norm_clip': 0.0,
    'batch_size': 64,
}