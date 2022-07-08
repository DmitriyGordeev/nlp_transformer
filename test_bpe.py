import unittest
from bpemb import BPEmb
import jsonlines
import numpy as np


class TestBPE(unittest.TestCase):

    def test_methods(self):
        bpemb_en = BPEmb(lang="en", dim=50)

        s = "Hey, wanna some coffee?"
        v = bpemb_en.encode(s)
        ids = bpemb_en.encode_ids(s)
        vecs = bpemb_en.vectors[ids]

        # emb = bpemb_en.emb

        out = bpemb_en.decode_ids(ids)
        pass


    def test_encode_billsumv3(self):
        bpemb_en = BPEmb(lang="en", dim=50)

        filepath = "data/summarizer_billsum_v3/ca_test_data_final_OFFICIAL.jsonl"
        data_tuples = []
        max_text_len = 0
        max_summary_len = 0
        with jsonlines.open(filepath) as f:
            print(f"Encoding text and summary strings ...")
            for line in f.iter():
                text = line['text']
                summary = line['summary']
                title = line['title']

                # encode text and summary with bpemb
                text_ids = bpemb_en.encode_ids(text)
                summary_ids = bpemb_en.encode_ids(summary)

                data_tuples.append((text_ids, summary_ids))
                if max_text_len < len(text_ids):
                    max_text_len = len(text_ids)
                if max_summary_len < len(summary_ids):
                    max_summary_len = len(summary_ids)

        pass


    def test_list_remove(self):
        L = [0, 2, 3, 10000, 3, 10000, 4, 10000, -1, 9]
        L = [x for x in L if x != 10000]

        pass





