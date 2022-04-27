import numpy
from collections import Counter
import re


class Tokenizer:
    def __init__(self, SOS=1, EOS=2):
        self.SOS = SOS
        self.EOS = EOS
        self.UNK = 0
        self.word2idx = None


    def assemble_vocab(self, text: str):
        prepared_text = self.cleanup(text)
        prepared_text = prepared_text.lower()

        text_without_dots = prepared_text.replace(".", "")
        words = text_without_dots.split(' ')

        vocab = Counter(words)      # create a dictionary
        vocab = sorted(vocab, key=vocab.get, reverse=True)

        # map words to unique indices
        # ind + 3 because 0, 1, 2 already occupied
        self.word2idx = {word: ind + 3 for ind, word in enumerate(vocab)}
        if '' in self.word2idx:
            del self.word2idx['']

        self.UNK = len(self.word2idx)    # todo: what if set it to 0 ?

        self.word2idx["<SOS>"] = self.SOS
        self.word2idx["<EOS>"] = self.EOS
        self.word2idx["<UNK>"] = self.UNK = len(self.word2idx)

        return self.get_encoded_data(prepared_text)


    def get_encoded_data(self, text: str):
        # sentences = text.split(".")
        # assert len(sentences) > 0
        # encoded_sentences = [0] * len(sentences)
        # max_observed_len = 0
        #
        # for i in range(len(sentences)):
        #     es = self.encode_sentence(sentences[i])
        #     if es is not None:
        #         if len(es) > max_observed_len:
        #             max_observed_len = len(es)
        #         encoded_sentences[i] = numpy.array(es)
        #
        # encoded_sentences = list(filter((0).__ne__, encoded_sentences))


        text = text.replace(". ", " ")
        text = re.sub("\s{2,}", " ", text)
        encoded_text = self.encode_sentence(text)
        encoded_text = encoded_text[1:-1]
        cursor = 0
        n_src = 10
        n_tgt = 10
        shift = 4

        data = list()
        while cursor < len(encoded_text) - (n_src + n_tgt):
            src = encoded_text[cursor:cursor + n_src]
            src.insert(0, self.SOS)
            src.append(self.EOS)

            tgt = encoded_text[cursor + n_src : cursor + n_src + n_tgt]
            tgt.insert(0, self.SOS)
            tgt.append(self.EOS)

            data.append([src, tgt])
            cursor += shift
            pass
        return data, self.word2idx


        # if len(encoded_sentences) == 0:
        #     raise ValueError("len(encoded_sentences) == 0")
        #
        # # Padding
        # padded_matrix = self.UNK * numpy.ones((len(encoded_sentences), max_observed_len), dtype=float)
        # for i in range(len(encoded_sentences)):
        #     padded_matrix[i, 0:len(encoded_sentences[i])] = encoded_sentences[i]
        #
        # # # rearrange sentences so senteces come with pairs (src, tgt)
        # # list_of_pairs_src_tgt = []
        # # for i in range(padded_matrix.shape[0] - 1):
        # #     list_of_pairs_src_tgt.append([padded_matrix[i,:], padded_matrix[i + 1,:]])
        # # return list_of_pairs_src_tgt, self.word2idx


    def encode_sentence(self, sentence: str):
        """
        :param sentence: 'Hello my name is George'
        :return: '[11, 21, 31, 4, 8]' each word is encoded according to word2idx dict
        """
        sentence = sentence.lower()
        sentence = sentence.strip()
        if sentence == "":
            return None
        words = sentence.split(" ")
        output = [0] * len(words)
        output.insert(0, self.SOS)

        for wi in range(len(words)):
            if words[wi] == "":
                continue
            if words[wi] not in self.word2idx:
                print(f"'{words[wi]}' is not found in word2idx")
            output[wi + 1] = self.word2idx[words[wi]]   # wi + 1 because we've added SOS manually
        output.append(self.EOS)
        return output


    def decode_sentence(self, indicies: list):
        words = []
        for idx in indicies:
            vocab_values = list(self.word2idx.values())
            if idx in vocab_values:
                word = list(self.word2idx.keys())[vocab_values.index(idx)]
                words.append(word)
            else:
                print (f"[debug][arr2text()] v = {idx}")
                words.append("<UNK>")
        return " ".join(words)


    @staticmethod
    def cleanup(text: str):
        text = text.replace('\n', "")
        text = text.replace('?', ".")
        text = text.replace('!', ".")
        text = text.replace(',', "")
        text = text.replace(':', "")
        text = text.replace(';', "")
        text = text.replace(".", ". ")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("\"", "")
        return text



