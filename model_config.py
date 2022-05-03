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
    'test_path': 'data/TheTimeMachine.txt',
    'val_path': 'data/saturn.txt',
    'seq_length': 128,
}

TransformerLanguageModelInfo = {
    'name': 'model1',
}