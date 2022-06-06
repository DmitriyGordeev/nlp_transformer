import math

import torch
import torch.nn as nn

from .pos_encoding import PositionalEncoding


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout_p: float,
    ):
        super().__init__()

        self.model_type = 'Transformer Language Model'
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
        
        self.transformer = nn.Transformer(
                                        d_model=d_model,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout_p,
                                        )

        self.linear = nn.Linear(
                            in_features=d_model,
                            out_features=num_tokens,
                            )
    
    def forward(
            self,
            src: list,
            tgt: list,
            tgt_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            ):

                src = self.embedding(src) * math.sqrt(self.d_model)
                tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
                src = self.positional_encoder(src)
                tgt = self.positional_encoder(tgt)
        
                src = src.permute(1,0,2)
                tgt = tgt.permute(1,0,2)

                out = self.transformer(
                                    src=src,
                                    tgt=tgt,
                                    tgt_mask=tgt_mask,
                                    src_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    )
                
                out = out.permute(1,0,2)        
                out = self.linear(out)
        
                return out

    def get_tgt_mask(self, size) -> torch.tensor:
        
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)
    
    def count_params(self):
        """ Counts number of parameters in the network exposed to gradient optimization """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_and_freeze_pretrained_embedding(self, tensor: torch.FloatTensor):
        """ :param tensor - FloatTensor of size (vocab, dim_model) - pretrained weights of embedding
            And freezes the layer to avoid training
        """
        self.embedding = nn.Embedding.from_pretrained(tensor)
        self.embedding.requires_grad_(False)

