from training_setup_interface import *
from classifier_model import *
import pandas
from data_loader import DatasetClassifierModel


class ClassiferSetup(TrainingSetup):
    def __init__(self, is_gpu: bool,
                 is_resume_mode: bool,
                 train_params: TrainParams,
                 model_params: ModelParams):
        super().__init__(is_gpu,
                         is_resume_mode,
                         train_params,
                         model_params)

        self.num_correct_epoch = 0
        self.num_comparisons_per_epoch = 0


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

        self.pretrained_embedding = self.tokenizer.load_pretrained_embedding(
            "pretrained_embedding_vocab/glove.6B.50d.top30K.txt",
            top_n=25000
        )

        self.word2idx = self.tokenizer.word2idx
        self.idx2word = self.tokenizer.idx2word
        self.word2idx_size = self.tokenizer.word2idx_size
        print(f"Vocab size {self.word2idx_size}")
        torch.save(self.word2idx, 'models/' + self.train_params.path_nm + '/vocab.pt')

        train_matrix, train_classes = self.prepare_data(train_path)
        self.train_dataset = DatasetClassifierModel(
            data_matrix=train_matrix,
            tgt_classes=train_classes
        )

        val_matrix, val_classes = self.prepare_data(val_path)
        self.val_dataset = DatasetClassifierModel(
            data_matrix=val_matrix,
            tgt_classes=val_classes
        )

        test_matrix, test_classes = self.prepare_data(test_path)
        self.test_dataset = DatasetClassifierModel(
            data_matrix=test_matrix,
            tgt_classes=test_classes
        )


    # TODO: move to DataLoader ?
    def prepare_data(self, filepath: str):
        """
        Input - dataframe with 2 columns (reviews and sentiment=('positive' or 'negative))
        Output - tuple = (matrix with encoded sequences, array of classes for each sequence (int))
        :param filepath - path to csv file
        """
        df = pandas.read_csv(filepath)
        vocab = self.tokenizer.word2idx
        max_review_len = 0

        start_token = vocab[model_constants.start_token]
        end_token = vocab[model_constants.end_token]
        pad_token = vocab[model_constants.pad_token]

        encoded_sequences = []
        tgt_classes = numpy.zeros(df.shape[0])

        for i in range(df.shape[0]):
            review = df.iloc[i, 0]
            seq = self.tokenizer.cleanup(data=review, tokenizer=TokenizerCollection.basic_english_by_word)
            if len(seq) > max_review_len:
                max_review_len = len(seq)

            enc_seq = self.tokenizer.encode_seq(seq)

            # surround sequence with start, end tokens:
            enc_seq.insert(0, start_token)
            enc_seq.append(end_token)
            encoded_sequences.append(numpy.array(enc_seq))       # todo: optimize this

            tgt_class = 1 if df.iloc[i, 1] == "positive" else 0
            tgt_classes[i] = tgt_class

        # Create matrix filled with padding tokens:
        data_matrix = numpy.zeros(shape=(df.shape[0], max_review_len + 2)) + pad_token      # +2 because we have sos and eos
        for i in range(len(encoded_sequences)):
            data_matrix[i, :len(encoded_sequences[i])] = encoded_sequences[i]
        return data_matrix, tgt_classes


    def setup_nn(self):
        self.nn_model = TransformerClassifierModel(num_tokens=self.word2idx_size,
                                                   d_model=self.model_params.d_model,
                                                   nhead=self.model_params.nhead,
                                                   num_encoder_layers=self.model_params.num_encoder_layers,
                                                   dim_feedforward=self.model_params.dim_feedforward,
                                                   dropout_p=self.model_params.dropout_p)

        # self.nn_model = model.TransformerLanguageModel(
        #     num_tokens=self.word2idx_size,
        #     d_model=self.model_params.d_model,
        #     nhead=self.model_params.nhead,
        #     num_encoder_layers=self.model_params.num_encoder_layers,
        #     num_decoder_layers=self.model_params.num_decoder_layers,
        #     dim_feedforward=self.model_params.dim_feedforward,
        #     dropout_p=self.model_params.dropout_p,
        # )


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
        src = batch[0].long().to(self.device)
        tgt = batch[1].long().to(self.device)

        pred = self.nn_model(src)

        correct = pred.argmax(axis=1) == tgt
        self.num_correct_epoch += correct.sum().item()
        self.num_comparisons_per_epoch += correct.size(0)

        loss = self.criterion(pred, tgt)

        # if print_enabled:
        #     # Print predicted sequence
        #     predicted_sequence = self.predict(src[0:1, :], max_length=self.train_params.inference_max_len)
        #
        #     # TODO: move from stdout to the logger
        #
        #     # decode src, tgt and prediction into human-readable string
        #     print("=================================================================")
        #     print(f"Predicted sequence, max_inference_len = {self.train_params.inference_max_len} : ")
        #     print(f"src  = {' '.join(self.tokenizer.decode_seq(src[0, :].view(-1).tolist()))}")
        #     print(f"tgt  = {' '.join(self.tokenizer.decode_seq(tgt_expected[0, :].view(-1).tolist()))}")
        #     print(f"pred = {' '.join(self.tokenizer.decode_seq(predicted_sequence))}")
        #     print("=================================================================")
        return pred, loss


    def AfterEpoch(self):
        print (self.num_correct_epoch)
        self.num_correct_epoch = 0



