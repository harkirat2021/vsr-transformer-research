import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl

from src.model_modules import *

#### VSR Models ####

""" VSR Transformer Encoder 1 """
class VSRTE1(pl.LightningModule):
    def __init__(self, name, c, h, w, embed_dim, n_head, h_dim, n_layers, dropout=0.5):
        super(VSRTE1, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, h_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.frame_encoder = FrameSeqEncoder(c, h, w, embed_dim)
        self.embed_dim = embed_dim
        self.frame_decoder = FrameSeqDecoder(c, h, w, embed_dim)

        self.name = name

    def set_src_mask(self, sz):
        self.src_mask = self.generate_square_subsequent_mask(sz)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src, src_mask):
        # Swap batch and sequence dimension
        src = torch.transpose(src, 0, 1)

        x = self.frame_encoder(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_mask)
        output = self.frame_decoder(output)

        # Residual connection
        output += src

        # Swap back batch and sequence dimension
        output = torch.transpose(output, 0, 1)

        return output
    
    ### Pytorch Lightning functions ###

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x, self.src_mask)
        loss = self.mse_loss(outputs[outputs.shape[0]//2], y[y.shape[0]//2])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x, self.src_mask)
        loss = self.mse_loss(outputs[outputs.shape[0]//2], y[y.shape[0]//2])
        self.log('valid_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer