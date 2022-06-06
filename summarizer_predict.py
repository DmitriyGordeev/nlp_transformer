import torch
from transformer_utils.model_constants import *
from transformer_utils.tokenizer import TokenizerLanguageModel, TokenizerCollection

# from tokenizer import TokenizerSummarizerModel, TokenizerCollection


def predict(
        model,
        word2idx: dict,
        text: str,
        device,
        max_length=20
):
    """ Infer sequence from input_sequence """
    tokenizer = TokenizerLanguageModel(
        pad_token=special_tokens.pad_token,
        start_token=special_tokens.start_token,
        end_token=special_tokens.end_token,
        unk_token=special_tokens.unk_token,
        pad_token_num=special_tokens.pad_token_num,
        start_token_num=special_tokens.start_token_num,
        end_token_num=special_tokens.end_token_num,
        unk_token_num=special_tokens.unk_token_num,
    )
    tokenizer.set_vocab(word2idx=word2idx)

    input_seq = tokenizer.cleanup(data=text, tokenizer=TokenizerCollection.basic_english_by_word)
    input_seq = [tokenizer.start_token_num] + tokenizer.encode_seq(input_seq)
    input_seq = torch.tensor([input_seq], dtype=torch.long, device=device)
    model.eval()
    y_input = torch.tensor([[tokenizer.start_token_num]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length)
        src_key_padding_mask = model.create_pad_mask(input_seq, tokenizer.pad_token_num)
        tgt_key_padding_mask = model.create_pad_mask(y_input, tokenizer.pad_token_num)

        pred = model(
            src=input_seq,
            tgt=y_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        top_num = 0
        esc_flag = True
        while esc_flag:
            next_item = pred.topk(4)[1].view(-1)[top_num].item()
            if next_item not in [special_tokens.start_token_num, special_tokens.pad_token_num,
                                 special_tokens.unk_token_num]:
                esc_flag = False
            top_num += 1
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    begin = 'Using the EAGLE suite of simulations, we demonstrate that both cold gas stripping {\it and} starvation of gas inflow play an important role in quenching satellite galaxies across a range of stellar and halo masses, M⋆ and M200. By quantifying the balance between gas inflows, outflows, and star formation rates, we show that even at z=2, only ≈30% of satellite galaxies are able to maintain equilibrium or grow their reservoir of cool gas - compared to ≈50% of central galaxies at this redshift. We find that the number of orbits completed by a satellite is a very good predictor of its quenching, even more so than the time since infall. On average, we show that intermediate-mass satellites with M⋆ between 109M⊙−1010M⊙ will be quenched at first pericenter in massive group environments, M200>1013.5M⊙; and will be quenched at second pericenter in less massive group environments, M200<1013.5M⊙. On average, more massive satellites (M⋆>1010M⊙) experience longer depletion time-scales, being quenched between first and second pericenters in massive groups; while in smaller group environments, just ≈30% will be quenched even after two orbits. Our results suggest that while starvation alone may be enough to slowly quench satellite galaxies, direct gas stripping, particularly at pericenters, is required to produce the short quenching time-scales exhibited in the simulation.'

    nn_model = torch.load('models/model1/model.pth', map_location=torch.device(device))
    vocab = torch.load('models/model1/vocab.pt', map_location=torch.device(device))

    continuation = predict(
        model=nn_model,
        word2idx=vocab,
        text=begin,
        max_length=20,
        device=device
    )
    print('begining:', begin)
    print('continuation:', continuation)


if __name__ == "__main__":
    main()
