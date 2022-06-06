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


TransformerSummarizerModelConfig = {
    'd_model': 64,
    'nhead': 16,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 1024,
    'dropout_p': 0.1,
}


TransformerSummarizerModelDataConfig = {
    
}


TransformerSummarizerModelInfo = {
    'name': 'model1',
}


TransformerSummarizerModelTrainConfig = {
    'epochs': 30,
    'learning_rate': 0.001,
    'inference_max_len': 10,
    'grad_norm_clip': 0.0,
    'batch_size': 2,
    'weight_decay': 0.0,        # L2 norm coeff: the bigger -> the less overfitting but the slower training
}


def save_model_config(directory, dict_param=None):
    """ Saves all settings to model_config.json file """
    json_dict = dict()
    if dict_param is None:      # manual
        json_dict["langModel"] = TransformerSummarizerModelConfig
        json_dict["data"] = TransformerSummarizerModelDataConfig
        json_dict["info"] = TransformerSummarizerModelInfo
        json_dict["train"] = TransformerSummarizerModelTrainConfig
    else:       # automatic
        json_dict["langModel"] = dict_param['TransformerSummarizerModelConfig']
        json_dict["data"] = dict_param['TransformerSummarizerModelDataConfig']
        json_dict["info"] = dict_param['TransformerSummarizerModelInfo']
        json_dict["train"] = dict_param['TransformerSummarizerModelTrainConfig']

    f = open(directory + "/model_config.json", "w")
    f.write(json.dumps(json_dict, indent=4))
    f.close()
