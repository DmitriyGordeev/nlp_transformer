from transformer_utils.training_setup_abstract import *
from transformer_utils.tokenizer import *
import transformer_utils.model_constants as model_constants
from transformer_utils.data_loader import DatasetLanguageModel
import transformer_utils.model as model


class LangModelSetup(TrainingSetup):
    def __init__(self, is_gpu: bool,
                 is_resume_mode: bool,
                 train_params: TrainParams,
                 model_params: ModelParams):
        super().__init__(is_gpu,
                         is_resume_mode,
                         train_params,
                         model_params)

    def load_data(
            self,
            train_path: str,
            test_path: str,
            val_path: str,
    ):
        """ Reads file, tokenize and prepare tensors to train """
        self.tokenizer = TokenizerLanguageModel(
            pad_token=model_constants.pad_token,
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            unk_token=model_constants.unk_token,
            pad_token_num=model_constants.pad_token_num,
            start_token_num=model_constants.start_token_num,
            end_token_num=model_constants.end_token_num,
            unk_token_num=model_constants.unk_token_num,
        )

        f = open(train_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        # self.tokenizer.assemble_vocab(text)

        # self.tokenizer.load_vocab_from_file("vocabs/10k.txt")
        self.pretrained_embedding = self.tokenizer.load_pretrained_embedding(
            "pretrained_embedding_vocab/glove.6B.50d.top30K.txt",
            top_n=25000
        )

        self.word2idx = self.tokenizer.word2idx
        self.idx2word = self.tokenizer.idx2word
        self.word2idx_size = self.tokenizer.word2idx_size
        self.train_data = self.tokenizer.encode_seq(text)
        f.close()

        print(f"Vocab size {self.word2idx_size}")

        torch.save(self.word2idx, 'models/' + self.train_params.path_nm + '/vocab.pt')

        f = open(test_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        self.test_data = self.tokenizer.encode_seq(text)
        f.close()

        f = open(val_path, "r", encoding="utf-8")
        text = f.read()
        text = self.tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
        self.val_data = self.tokenizer.encode_seq(text)
        f.close()

        self.train_dataset = DatasetLanguageModel(
            data=self.train_data,
            sequence_length=self.train_params.seq_length,
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            pad_token=model_constants.pad_token,
            vocab=self.word2idx,
        )
        self.test_dataset = DatasetLanguageModel(
            data=self.test_data,
            sequence_length=self.train_params.seq_length,
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            pad_token=model_constants.pad_token,
            vocab=self.word2idx,
        )
        self.val_dataset = DatasetLanguageModel(
            data=self.val_data,
            sequence_length=self.train_params.seq_length,
            start_token=model_constants.start_token,
            end_token=model_constants.end_token,
            pad_token=model_constants.pad_token,
            vocab=self.word2idx,
        )


    def setup_nn(self):
        self.nn_model = model.TransformerLanguageModel(
            num_tokens=self.word2idx_size,
            d_model=self.model_params.d_model,
            nhead=self.model_params.nhead,
            num_encoder_layers=self.model_params.num_encoder_layers,
            num_decoder_layers=self.model_params.num_decoder_layers,
            dim_feedforward=self.model_params.dim_feedforward,
            dropout_p=self.model_params.dropout_p,
        )


    def setup_optimizers(self):
        self.optimizer = top.Adam(self.nn_model.parameters(),
                                  lr=self.train_params.learning_rate,
                                  weight_decay=self.train_params.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                    patience=5,
                                                                    threshold=0.001,
                                                                    factor=0.5)


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



