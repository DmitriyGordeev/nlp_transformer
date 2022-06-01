import math
import torch
import torch.nn as nn
from pos_encoding import PositionalEncoding


class TransformerClassifierModel(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            dropout_p: float,
    ):
        super().__init__()

        self.model_type = 'Transformer Classification Model'
        self.d_model = d_model

        self.positional_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout_p,
            max_len=5000
        )

        self.embedding = nn.Embedding(
            num_embeddings=num_tokens,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.classifier = nn.Linear(d_model, 2)


    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)

        out = self.transformer_encoder(src)
        out = out.mean(dim=1)
        out = self.classifier(out)
        return out


    def count_params(self):
        """ Counts number of parameters in the network exposed to gradient optimization """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
