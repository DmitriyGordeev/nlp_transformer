TransformerLanguageModelConfig = {
    'd_model': 8,
    'nhead': 4,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1,
    'dim_feedforward': 1024,
    'dropout_p': 0.01,
}

TransformerLanguageModelDataConfig = {
    'train_path': 'data/JV13.txt',
    'test_path': 'data/JV13_part1.txt',
    'val_path': 'data/JV13_part2.txt',
    'seq_length': 10,
}

TransformerLanguageModelInfo = {
    'name': 'model1',
}

TransformerLanguageModelTrainConfig = {
    'epochs': 22,
    'learning_rate': 0.0000075,
    'inference_max_len': 10,
    'grad_norm_clip': 0.0,
    'batch_size': 128,
    'weight_decay': 0.0         # L2 norm coeff: the bigger -> the less overfitting but the slower training
}