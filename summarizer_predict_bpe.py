from bpemb import BPEmb
import torch
from pathlib import Path


def main():
    device = "cpu"

    # Load bpemb pre-trained model
    bpe_embedding_dim = 50
    bpemb_en = BPEmb(lang="en", dim=bpe_embedding_dim, cache_dir=Path("bpe_model/"))
    pad_token_index = 10000     # todo: should be equal to the value assigned in summarizer_setup.py
                                #   ( self.pad_token_index = embedding_vectors.shape[0] - 1 )

    # Load transformer model
    nn_model = torch.load('models/model1/checkpoints/model.182.pth', map_location=torch.device(device))
    nn_model.eval()

    input_string = ""

    input_ids = torch.LongTensor(bpemb_en.encode_ids(input_string)).unsqueeze(0)     # todo: long tensor ?

    # Predicting loop
    max_length = 100
    y_input = torch.tensor([[bpemb_en.BOS]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = nn_model.get_tgt_mask(y_input.size(1)).to(device)

        pred = nn_model(input_ids, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # ignore <pad> tokens
        if next_item == pad_token_index:
            continue

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == bpemb_en.EOS:
            break

    output_ids = y_input.view(-1).tolist()
    output_string = bpemb_en.decode_ids(output_ids)

    print ("Input:", input_string)
    print ("Predicted: ", output_string)


if __name__ == "__main__":
    main()







