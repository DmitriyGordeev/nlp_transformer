import torch
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection
import transformer_utils.model_constants as model_constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(
    model,
    word2idx: dict,
    text: str,
    max_length=10
):
    """ Infer sequence from input_sequence """
    tokenizer = TokenizerLanguageModel(
                                    pad_token=model_constants.pad_token,
                                    start_token=model_constants.start_token,
                                    end_token=model_constants.end_token,
                                    unk_token=model_constants.unk_token,
                                    pad_token_num=model_constants.pad_token_num,
                                    start_token_num=model_constants.start_token_num,
                                    end_token_num=model_constants.end_token_num,
                                    unk_token_num=model_constants.unk_token_num,
                                    )
    tokenizer.set_vocab(word2idx=word2idx)

    input_seq = tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
    input_seq = [tokenizer.start_token_num] + tokenizer.encode_seq(input_seq)
    input_seq = torch.tensor([input_seq], dtype=torch.long, device=device)
    model.eval()
    y_input = torch.tensor([[tokenizer.start_token_num]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_seq, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == tokenizer.end_token_num:
            break
        
    ans = y_input.view(-1).tolist()[1:]

    ans = tokenizer.decode_seq(ans)
    ans = tokenizer.buildup(ans)

    return ans


if __name__ == "__main__":

    begin = 'Where are my cookies '

    nn_model = torch.load('models/model1/checkpoints/model.180.pth', map_location=torch.device(device))
    vocab = torch.load('models/model1/vocab.pt', map_location=torch.device(device))

    continuation = predict(
                        model=nn_model,
                        word2idx=vocab,
                        text=begin,
                        max_length=10,
                        )
    print('begining:', begin)
    print('continuation:', continuation)