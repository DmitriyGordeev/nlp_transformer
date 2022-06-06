import torch
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection
import transformer_utils.model_constants as model_constants


def get_tgt_mask(size) -> torch.tensor:
    mask = torch.tril(torch.ones(size, size) == 1)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == "__main__":
    """ Converts model to a jit file to run without python class """

    input_text = 'he said it'

    model = torch.load('models/model1/checkpoints/model.125.pth')
    vocab = torch.load('models/model1/vocab.pt')

    device = "cpu"

    model = model.to(device)

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
    tokenizer.set_vocab(word2idx=vocab)

    input_seq = tokenizer.cleanup(data=input_text, tokenizer=TokenizerCollection.basic_english_by_word)
    input_seq = [tokenizer.start_token_num] + tokenizer.encode_seq(input_seq)
    input_seq = torch.tensor([input_seq], dtype=torch.long, device=device)
    model.eval()

    y_input = torch.tensor([[tokenizer.start_token_num]], dtype=torch.long, device=device)

    tgt_mask = get_tgt_mask(y_input.size(1))

    # Specify example input to the network and save
    model = torch.jit.trace(model, example_inputs=(input_seq, y_input, tgt_mask))
    torch.jit.save(model,'model.jit')

    jit_model = torch.jit.load('model.jit')

    max_inference_len = 10

    for _ in range(max_inference_len):
        # Get source mask
        tgt_mask = get_tgt_mask(y_input.size(1)).to(device)

        pred = jit_model(input_seq, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == tokenizer.end_token_num:
            break

    continuation = y_input.view(-1).tolist()[1:]
    continuation = tokenizer.decode_seq(continuation)
    continuation = tokenizer.buildup(continuation)

    print('input:', input_text)
    print('continuation:', continuation)





