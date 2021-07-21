import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl

#### VSR Models ####

""" VSR Transformer Encoder 1 """
class VSRTE1(pl.LightningModule):
    def __init__(self, c, h, w, embed_dim, n_head, h_dim, n_layers, dropout=0.5):
        super(VSRTE1, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_head, h_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.frame_encoder = FrameSeqEncoder(c, h, w, embed_dim)
        self.embed_dim = embed_dim
        self.frame_decoder = FrameSeqDecoder(c, h, w, embed_dim)

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
        self.log('val_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer


#### Modules ####

class BicubicResize(nn.Module):
    KERNEL_WIDTH = 4.0

    def __init__(self, scale, im_size, cuda=True):
        super(BicubicResize, self).__init__()
        self.input_size = im_size
        self.gpu = cuda
        self.set_scale(scale)

    def set_scale(self, scale):
        self.scale = scale
        self.output_size = self.deriveSizeFromScale(self.input_size)

        self.weights = []
        self.indices = []

        for k in range(2):
            w, ind = self.contributions(self.input_size[k], self.output_size[k],
                                   self.scale[k], self.cubic, self.KERNEL_WIDTH)
            self.weights.append(w)
            self.indices.append(ind)

    def deriveSizeFromScale(self, input_size ):
        output_shape = [int(ceil(self.scale[0] * input_size[0])),
                        int(ceil(self.scale[1] * input_size[1])),
                        ]

        return output_shape

    def cubic(self, x):
        absx = np.abs(x)
        absx2 = absx*absx
        absx3 = absx2*absx
        f = ((1.5*absx3 - 2.5*absx2 + 1)*(absx <= 1)) + ((-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) * (absx <= 2)))

        return f

    def contributions(self, in_length, out_length, scale, kernel, k_width):
        if scale < 1:
            h = lambda x: scale * kernel(scale * x)
            kernel_width = 1.0 * k_width / scale
        else:
            h = kernel
            kernel_width = k_width
        x = np.arange(1, out_length+1).astype(np.float64)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = np.floor(u - kernel_width / 2)
        P = int(ceil(kernel_width)) + 2
        ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
        indices = ind.astype(np.int32)
        weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
        weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
        aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
        indices = aux[np.mod(indices, aux.size)]
        ind2store = np.nonzero(np.any(weights, axis=0))
        weights = torch.from_numpy(weights[:, ind2store]).float()
        indices = torch.from_numpy(np.squeeze(indices[:, ind2store])).contiguous().long()

        if self.gpu:
            weights = weights.cuda()
            indices = indices.cuda()

        return Variable(weights), Variable(indices)

    def resizeAlongDim(self, inimg, weights, indices, dim):
        w_shape = weights.size()
        idx_size = indices.size()
        indices = indices.view(-1)

        if dim==0:
            im_slice = torch.index_select(inimg, 1, indices)
            im_slice = im_slice.view(im_slice.size(0), idx_size[0], idx_size[1], -1, im_slice.size(3))
            im_slice = torch.transpose(im_slice, 2, 3)
            w = weights.view(1, w_shape[0], 1, 1, -1)
        elif dim==1:
            im_slice = torch.index_select(inimg, 2, indices)
            im_slice = im_slice.view(im_slice.size(0), -1, idx_size[0], idx_size[1], im_slice.size(3))
            w = weights.view(1, 1, w_shape[0], 1, -1)

        outimg = torch.squeeze(torch.matmul(w, im_slice), dim=3)

        return outimg

    def forward(self, x):
        scale = self.scale
        scale_np = np.array(self.scale)
        order = np.argsort(scale_np)

        if self.train:
            weights = self.weights
            indices = self.indices
        else:
            weights = []
            indices = []

            for k in range(2):
                input_size = x.size()[2:]
                output_size = self.deriveSizeFromScale(input_size)
                w, ind = self.contributions(input_size[k], output_size[k],
                                   self.scale[k], self.cubic, self.KERNEL_WIDTH)
                weights.append(w)
                indices.append(ind)

        x = torch.squeeze(torch.transpose(torch.unsqueeze(x, dim=4), 1, 4), dim=1)

        for k in range(2):
            dim = order[k]
            x = self.resizeAlongDim(x, weights[dim], indices[dim], dim)

        x = torch.squeeze(torch.transpose(torch.unsqueeze(x, dim=1), 1, 4), dim=4)

        return x

class RGB2YCBCR(nn.Module):
    def __init__(self):
        super(RGB2YCBCR, self).__init__()
        ycbcr_from_rgb = torch.from_numpy(np.array([[    65.481,   128.553,    24.966],
                            [   -37.797,   -74.203,   112.0  ],
                            [   112.0  ,   -93.786,   -18.214]])).float().view(3, 3)*(1/255.0)
        sum_const = torch.from_numpy(np.array([    16,   128,    128])).float().view(1, 3, 1, 1)*(1/255.0)
        self.register_buffer('ycbcr_from_rgb', ycbcr_from_rgb)
        self.register_buffer('sum_const', sum_const)

    def forward(self, x):
        or_size = x.size()
        x = torch.matmul(self.ycbcr_from_rgb, x.view(x.size(0), 3, -1))
        x = x.view(or_size)+self.sum_const

        return x

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
    
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            x = self.frame_decoder(x)
            y_seq.append(x)
        return torch.stack(y_seq, dim=0)

"""  """
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

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.frame_encoder(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.frame_decoder(output)
        return output