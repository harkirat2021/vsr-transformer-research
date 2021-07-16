import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

""" Encode position in sequence """
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

""" Frame Encoder """
class FrameEncoder(nn.Module):
    def __init__(self, c, h, w, emsize):
        super(FrameEncoder, self).__init__()
        self.emsize = emsize
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear((h // 4) * (w // 4) * 8, emsize)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

""" Frame Decoder """
class FrameDecoder(nn.Module):
    def __init__(self, c, h, w, emsize):
        super(FrameDecoder, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.fc1 = nn.Linear(emsize, h // 4 * w // 4 * 8)
        self.conv1 = nn.Conv2d(8, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, c, 3, padding=1)

        self.convtrans2 = nn.ConvTranspose2d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.convtrans1 = nn.ConvTranspose2d(in_channels=12, out_channels=c, kernel_size=3, stride=2, padding=1)

        self.up_sample = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        x = self.fc1(x).reshape(x.shape[0], 8, self.h // 4, self.w // 4)
        x = F.relu(self.conv1(self.up_sample(x)))
        x = torch.sigmoid(self.conv2(self.up_sample(x)))
        return x

""" Frame Sequence Encoder """
class FrameSeqEncoder(nn.Module):
    def __init__(self, c, h, w, emsize):
        super(FrameSeqEncoder, self).__init__()
        self.emsize = emsize
        self.frame_encoder = FrameEncoder(c, h, w, emsize)
    
    def init_weights(self, initrange):
        self.frame_encoder.init_weights(initrange)
    
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            x = self.frame_encoder(x)
            y_seq.append(x)
        return torch.stack(y_seq, dim=0)

""" Frame Sequence Decoder """
class FrameSeqDecoder(nn.Module):
    def __init__(self, c, h, w, emsize):
        super(FrameSeqDecoder, self).__init__()
        self.frame_decoder = FrameDecoder(c, h, w, emsize)
    
    def init_weights(self, initrange):
        self.frame_decoder.init_weights(initrange)
    
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            x = self.frame_decoder(x)
            y_seq.append(x)
        return torch.stack(y_seq, dim=0)
    
class FrameAutoencoder(nn.Module):
    def __init__(self, c, h, w, e):
        super(FrameAutoencoder, self).__init__()
        self.encoder = FrameEncoder(c, h, w, e)
        self.decoder = FrameDecoder(c, h, w, e)
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

class FrameSeqAutoencoder(nn.Module):
    def __init__(self, c, h, w, e):
        super(FrameSeqAutoencoder, self).__init__()
        self.encoder = FrameSeqEncoder(c, h, w, e)
        self.decoder = FrameSeqDecoder(c, h, w, e)
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

""" Transformer """
class TransformerModel(nn.Module):
    def __init__(self, c, h, w, embed_dim, n_head, h_dim, n_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, h_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.frame_encoder = FrameSeqEncoder(c, h, w, embed_dim)
        self.embed_dim = embed_dim
        self.frame_decoder = FrameSeqDecoder(c, h, w, embed_dim)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.frame_encoder.init_weights(initrange)
        #self.frame_decoder.init_weights(initrange)

    def forward(self, src, src_mask):
        src = self.frame_encoder(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.frame_decoder(output)
        return output