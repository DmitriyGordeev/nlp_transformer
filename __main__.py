import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tokenizer import Tokenizer
from torch.nn.functional import one_hot
import data_generator
import model


def train_loop(__model, __opt, __loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    __model.train()
    total_loss = 0

    for batch in dataloader:
        X, y = batch[:, 0], batch[:, 1]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]      # TODO: should we store <EOS> here -> For now it's there!
        y_expected = y[:,1:]

        # Get mask to mask out the next words.
        sequence_length = y_input.size(1)
        tgt_mask = __model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = __model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)

        # loss = __loss_fn(pred, y_expected)
        loss = __loss_fn(pred, y_expected)

        __opt.zero_grad()
        loss.backward()
        __opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(__model, __loss_fn, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    __model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = __model.get_tgt_mask(sequence_length).to(device)
            tgt_pad_mask = __model.create_pad_mask(y_input, pad_token=0)

            # Standard training except we pass in y_input and src_mask
            pred = __model(X, y_input, tgt_mask, tgt_pad_mask=tgt_pad_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = __loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(__model, __opt, __loss_fn, __train_dataloader, __val_dataloader, epochs):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)

        train_loss = train_loop(__model, __opt, __loss_fn, __train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(__model, __loss_fn, __val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

    return train_loss_list, validation_loss_list


def predict(__model, input_sequence, max_length=80, SOS_token=1, EOS_token=2):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    __model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = __model.get_tgt_mask(y_input.size(1)).to(device)

        pred = __model(input_sequence, y_input, tgt_mask)

        # next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = pred[-1, :, :].argmax().item()  # num of the highest proba on the last dimension
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()


if __name__ == "__main__":

    tokenizer = Tokenizer()
    with open("space.txt", "r") as f:
        text = f.read()
        sentences, vocab = tokenizer.assemble_vocab(text)
    vocab_size = len(list(vocab.keys()))

    train_portion = 0.7
    num_train_samples = int(len(sentences) * train_portion)

    train_data = sentences[0:num_train_samples]
    val_data  = sentences[num_train_samples:]

    batch_size = 64
    learning_rate = 0.0001      # bigger lr gives better results ?
    weight_decay = 0.001       # TODO: how it works ?
    epochs = 1000

    # TODO: распечатать датасет как пары
    # TODO: calculate number of NN's parameters

    train_dataloader = data_generator.batchify_data(train_data, batch_size=batch_size)   # (list) element is a ndarray (batch_size, 2, sequence_len)
    val_dataloader = data_generator.batchify_data(val_data, batch_size=batch_size)       # (list) element is a ndarray (batch_size, 2, sequence_len)


    # 2. Setup training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    NNModel = model.Transformer(
        num_tokens=vocab_size, dim_model=64, num_heads=8, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.01
    ).to(device)

    opt = torch.optim.Adam(NNModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, validation_loss_list = fit(NNModel, opt, loss_fn, train_dataloader, val_dataloader, epochs)

    # todo: add plot

    examples = [
        torch.tensor([tokenizer.encode_sentence("it was hypothesized that")], dtype=torch.long, device=device)
    ]

    # TODO: set of test samples:
    for idx, example in enumerate(examples):
        result = predict(NNModel, example, max_length=10)
        print(f"Example {idx}")
        print(f"Input = {tokenizer.decode_sentence(example.view(-1).tolist())}")
        print(f"Continuation = {tokenizer.decode_sentence(result)}")
        print()
