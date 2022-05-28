import json
import numpy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


TransformerLanguageModelConfig = {
    'd_model': 50,
    'nhead': 10,
    'num_encoder_layers': 8,
    'num_decoder_layers': 8,
    'dim_feedforward': 8096,
    'dropout_p': 0.1,
}

TransformerLanguageModelDataConfig = {
    'train_path': 'data/concat.txt',
    'test_path': 'data/JV13_part1.txt',
    'val_path': 'data/TheTimeMachine.txt',
    'seq_length': 50,
}

TransformerLanguageModelInfo = {
    'name': 'model1',
}

TransformerLanguageModelTrainConfig = {
    'epochs': 100,
    'learning_rate': 0.0000075,
    'inference_max_len': 50,
    'grad_norm_clip': 0.0,
    'batch_size': 48,
    'weight_decay': 0.0         # L2 norm coeff: the bigger -> the less overfitting but the slower training
}


def save_model_config(directory, dict_param=None):
    """ Saves all settings to model_config.json file """
    json_dict = dict()
    if dict_param is None:      # manual
        json_dict["langModel"] = TransformerLanguageModelConfig
        json_dict["data"] = TransformerLanguageModelDataConfig
        json_dict["info"] = TransformerLanguageModelInfo
        json_dict["train"] = TransformerLanguageModelTrainConfig
    else:       # automatic
        json_dict["langModel"] = dict_param['TransformerLanguageModelConfig']
        json_dict["data"] = dict_param['TransformerLanguageModelDataConfig']
        json_dict["info"] = dict_param['TransformerLanguageModelInfo']
        json_dict["train"] = dict_param['TransformerLanguageModelTrainConfig']

    f = open(directory + "/model_config.json", "w")
    f.write(json.dumps(json_dict, indent=4))
    f.close()
