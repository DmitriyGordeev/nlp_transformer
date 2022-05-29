from training_setup_interface import *


class LangModelSetup(TrainingSetup):
    def __init__(self, is_gpu: bool, is_resume_mode: bool, train_params: TrainParams, model_params: ModelParams):
        super().__init__(is_gpu,
                         is_resume_mode,
                         train_params,
                         model_params)

    def nn_forward(self, batch, print_enabled=False):
        src = batch.to(self.device)
        tgt_input = batch[:, :-1].to(self.device)
        tgt_expected = batch[:, 1:].to(self.device)

        # Get mask to mask out the next words
        sequence_length = tgt_input.size(1)
        tgt_mask = self.nn_model.get_tgt_mask(sequence_length).to(self.device)

        # Standard training except we pass in y_input and tgt_mask
        pred = self.nn_model(src, tgt_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        tgt_expected = tgt_expected.type(torch.int64)

        loss = self.criterion(pred, tgt_expected)

        if print_enabled:
            # Print predicted sequence
            predicted_sequence = self.predict(src[0:1, :], max_length=self.train_params.inference_max_len)

            # TODO: move from stdout to the logger

            # decode src, tgt and prediction into human-readable string
            print("=================================================================")
            print(f"Predicted sequence, max_inference_len = {self.train_params.inference_max_len} : ")
            print(f"src  = {' '.join(self.tokenizer.decode_seq(src[0, :].view(-1).tolist()))}")
            print(f"tgt  = {' '.join(self.tokenizer.decode_seq(tgt_expected[0, :].view(-1).tolist()))}")
            print(f"pred = {' '.join(self.tokenizer.decode_seq(predicted_sequence))}")
            print("=================================================================")

        return pred, loss
