import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl

from src.model_modules import *

#### VSR Models ####

class VSRTE1(pl.LightningModule):
    def __init__(self, name, t, c, h, w, embed_dim, n_head, h_dim, n_layers, dropout=0.5):
        super(VSRTE1, self).__init__()
        self.model_type = 'Transformer'
        self.name = name

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, h_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.frame_encoder = FrameSeqEncoder(c, h, w, embed_dim)
        self.frame_decoder = FrameSeqDecoder(c, h, w, embed_dim)

        self.upsample = UpsampleSeqLayer(t, c, 3)

        # Mask for now - sequence doesn't really make sense here does it?
        self.src_mask = self.generate_empty_mask(t)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Swap batch and sequence dimension
        src = torch.transpose(src, 0, 1)

        # Forwad pass
        x = self.frame_encoder(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = self.frame_decoder(x)
        y = self.upsample(x)

        # Swap back batch and sequence dimension
        y = torch.transpose(y, 0, 1)

        return y
    
    ### Pytorch Lightning functions ###

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x, self.src_mask)
        # Compute loss between all, but first and last frame
        loss = self.mse_loss(outputs[1:-1], y[1:-1])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x, self.src_mask)
        loss = self.mse_loss(outputs[1:-1], y[1:-1])
        self.log('valid_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer


""" VSR Seq Autoencoder """
class VSRSA1(pl.LightningModule):
    def __init__(self, t, c, h, w, emsize, k, n_hidden, n_layers):
        super(VSRSA1, self).__init__()
        self.model_type = 'Autoencoder'
        self.name = name

        self.frame_encoder = FrameSeqEncoder(c, h, w, emsize)
        self.frame_decoder = FrameSeqDecoder(c, h, w, emsize)
        self.embedding_seq_transform = EmbeddingSeqTransform(t, emsize, k, n_hidden, n_layers)
        self.upsample = UpsampleSeqLayer(t, c, 3)

    """ (batch_dim, time_dim, channel_dim, height_dim, width_dim) -> (batch_dim, time_dim, channel_dim, height_dim, width_dim) """
    def forward(self, x):
        # Swap batch and sequence dimension
        x = torch.transpose(x, 0, 1)

        # Encode
        x = self.frame_encoder(x)

        # Swap back batch and sequence dimension
        x = torch.transpose(x, 0, 1)

        # Embed transform
        x = self.embedding_seq_transform(x)

        # Swap batch and sequence dimension
        x = torch.transpose(x, 0, 1)

        # Decode
        x = self.frame_decoder(x)

        # Upsample
        y = self.upsample(x)

        # Swap back batch and sequence dimension
        y = torch.transpose(y, 0, 1)

        return y
    
    ### Pytorch Lightning functions ###

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x, self.src_mask)
        # Compute loss between all, but first and last frame
        loss = self.mse_loss(outputs[1:-1], y[1:-1])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x, self.src_mask)
        loss = self.mse_loss(outputs[1:-1], y[1:-1])
        self.log('valid_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer