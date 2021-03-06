from math import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

# WARNING: This is for small numbers, otherwise, it will be very slow.
# A function to calculate all prime factors of
# a given number n
def primeFactors(n):
    factors = {}
    count = 0

    # Print the number of two's that divide n
    while n % 2 == 0:
        count += 1
        n = int(n / 2)

    if count > 0:
        factors[2] = count

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(sqrt(n))+1,2):
        count = 0
        # while i divides n , print i ad divide n
        while n % i== 0:
            count += 1
            n = int(n / i)

        if count > 0:
            factors[i] = count

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        factors[n] = 1

    return factors



class UpsampleLayer(nn.Module):
    def __init__(self, n_features, k, factor=2, act_class=nn.ReLU,
                 act_params={'inplace': True}):
        super(UpsampleLayer, self).__init__()
        self.n_factors  = 0
        self.n_features = n_features
        self.k          = k
        self.pad        = int(k//2)
        self.act_class  = act_class
        self.act_params = act_params

        factors = primeFactors(factor)

        for n, i in factors.items():
            for j in range(i):
                self.add_factor(n)

    def add_factor(self, n):
        self.n_factors += 1

        self.add_module('upsample_%d'%self.n_factors,
            nn.Sequential(
                nn.Conv2d(self.n_features, self.n_features*(n**2),
                          self.k, padding=self.pad),
                nn.PixelShuffle(n),
                self.act_class(**self.act_params),
            )
        )

    def forward(self, x):
        for i in range(1, self.n_factors+1):
            x = getattr(self, 'upsample_%d'%i)(x)

        return x

class UpsampleSeqLayer(nn.Module):
    def __init__(self, seq_len, n_features, k=3, factor=2, act_class=nn.ReLU, act_params={'inplace': True}):
        super(UpsampleSeqLayer, self).__init__()

        self.seq_len = seq_len
        self.upsample = UpsampleLayer(n_features, k, factor=factor, act_class=act_class, act_params=act_params)

    # Expects (seq, batch, channels, height, width)
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            y_seq.append(self.upsample(x))

        return torch.stack(y_seq, dim=0)

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

""" Transform a sequence of embeddings """
class EmbeddingSeqTransform(nn.Module):
    def __init__(self, t, emsize, n_hidden, n_layers, k=3):
        super(EmbeddingSeqTransform, self).__init__()
        self.emsize = emsize

        self.conv_start = nn.Conv1d(t, n_hidden, k, padding=k//2)
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_hidden, n_hidden, k, padding=k//2),
                    nn.Conv1d(n_hidden, n_hidden, k, padding=k//2)
                ) for i in range(n_layers)
            ]
        )
        self.conv_final = nn.Conv1d(n_hidden, t, k, padding=k//2)
    
    """ (batch_dim, time_dim, embed_dim) -> (batch_dim, time_dim, embed_dim) """
    def forward(self, x):
        x = F.relu(self.conv_start(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        y = F.relu(self.conv_final(x))
        return y

""" Frame Encoder """
class FrameEncoder(nn.Module):
    def __init__(self, c, h, w, emsize, n_hidden, n_layers, s):
        super(FrameEncoder, self).__init__()
        self.emsize = emsize
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=n_hidden, kernel_size=3, stride=s, padding=1)
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(n_hidden, n_hidden, 3, stride=s, padding=1)
                ) for i in range(n_layers)
            ]
        )
        self.conv2 = nn.Conv2d(in_channels=n_hidden, out_channels=8, kernel_size=3, stride=s, padding=1)
        self.fc1 = nn.Linear((h // (s ** (n_layers + 2))) * (w // (s ** (n_layers + 2))) * 8, emsize) if h > (
                    s ** (n_layers + 2)) else nn.Linear(8, emsize)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

""" Frame Decoder """
#### below are original codes!!!!!
'''class FrameDecoder(nn.Module):
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
'''

class FrameDecoder(nn.Module):
    def __init__(self, c, h, w, emsize, n_hidden, n_layers, s):
        super(FrameDecoder, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.s = s
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        # NOTE: should be divisible
        self.f1_dim = n_hidden * (w * h) // (4 * 4)
        self.fc1 = nn.Linear(emsize, self.f1_dim)

        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(n_hidden, n_hidden, 3, padding=1)
                ) for i in range(n_layers)
            ]
        )
        self.conv2 = nn.Conv2d(n_hidden, c, 3, padding=1)
        self.up_sample = nn.Upsample(scale_factor=s)

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(x.shape[0], self.n_hidden, self.h // 4, self.w // 4)
        for i, layer in enumerate(self.layers):
            if i < 1:
                x = F.relu(layer(self.up_sample(x)))
            else:
                x = F.relu(layer(x))

        x = torch.sigmoid(self.conv2(self.up_sample(x)))
        return x
        
""" Frame Sequence Encoder """
class FrameSeqEncoder(nn.Module):
    def __init__(self, c, h, w, emsize, n_hidden, n_layers, s):
        super(FrameSeqEncoder, self).__init__()
        self.emsize = emsize
        self.frame_encoder = FrameEncoder(c, h, w, emsize, n_hidden, n_layers, s)
    
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            x = self.frame_encoder(x)
            y_seq.append(x)
        return torch.stack(y_seq, dim=0)

""" Frame Sequence Decoder """
class FrameSeqDecoder(nn.Module):
    def __init__(self, c, h, w, emsize, n_hidden, n_layers, s):
        super(FrameSeqDecoder, self).__init__()
        self.frame_decoder = FrameDecoder(c, h, w, emsize, n_hidden, n_layers, s)
    
    def forward(self, x_seq):
        y_seq = []
        for x in x_seq:
            x = self.frame_decoder(x)
            y_seq.append(x)
        return torch.stack(y_seq, dim=0)

class PatchEncode(nn.Module):
    def __init__(self, f, c, h, w, emsize, p):
        super(PatchEncode, self).__init__()
        self.emsize = emsize
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=emsize, kernel_size=(f, p, p), stride=p, padding=0)

    def forward(self, x):
        # Swap channels and frames so we treat it as 3D
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        return x
        
class PatchDecode(nn.Module):
    def __init__(self, f, c, h, w, emsize, n_hidden, n_layers):
        super(PatchDecode, self).__init__()
        self.emsize = emsize
        self.n_hidden = n_hidden
        self.f = f
        self.c = c
        self.h = h
        self.w = w
        
        # TODO - temp for just this instance of the model
        self.fc1 = nn.Linear(16 * emsize, f * c * w * h)
        self.conv1 = nn.Conv3d(c, n_hidden, 3, padding=1)
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(n_hidden, n_hidden, 3, padding=1)
                ) for i in range(n_layers)
            ]
        )
        self.conv2 = nn.Conv3d(n_hidden, c, 3, padding=1)

    def forward(self, x):
        # Swap channels and frames so we treat it as 3D
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.reshape(x.shape[0], self.c, self.f, self.h, self.w)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Swap channels and frames back
        x = torch.transpose(x, 1, 2)

        return x

class ConvTransform(nn.Module):
    def __init__(self, c, n_hidden, n_layers):
        super(ConvTransform, self).__init__()

        self.conv1  = nn.Conv3d(c, n_hidden, 3, padding=1)
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(n_hidden, n_hidden, 3, padding=1)
                ) for i in range(n_layers)
            ]
        )
        self.conv2 = nn.Conv3d(n_hidden, c, 3, padding=1)

    def forward(self, x):
        # Swap channels and frames so we treat it as 3D
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.conv1(x))
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        x = F.sigmoid(self.conv2(x))
        x = torch.transpose(x, 1, 2)
        return x
        
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
        src = self.frame_encoder(src) * sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.frame_decoder(output)
        return output



def count_params(model):
    return sum(p.numel() for p in model.parameters())
    
""" Residual BottleNeck Unit """
class ResBottleNeckBlock(nn.Module):
    def __init__(self, n_channels, exp_channels):
        super().__init__()
        self.conv_1x1_1 = nn.Conv2d(n_channels, exp_channels, kernel_size=1)
        self.batch_norm_1 = nn.BatchNorm2d(exp_channels)

        self.conv_3x3_1 = nn.Conv2d(exp_channels, exp_channels, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(exp_channels)

        self.conv_3x3_2 = nn.Conv2d(exp_channels, exp_channels, kernel_size=3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(exp_channels)

        self.conv_1x1_2 = nn.Conv2d(exp_channels, n_channels, kernel_size=1)
        self.batch_norm_4 = nn.BatchNorm2d(n_channels)
    
    def forward(self, x):
        x_in = x.clone()
        x = self.batch_norm_1(F.relu(self.conv_1x1_1(x)))
        x = self.batch_norm_2(F.relu(self.conv_3x3_1(x)))
        x = self.batch_norm_3(F.relu(self.conv_3x3_2(x)))
        x = self.batch_norm_4(F.relu(self.conv_1x1_2(x)))
        return x + x_in

""" Local Attention Unit """
class TorchLocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW):
        super(TorchLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW

    @staticmethod
    def f_similar(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = x_theta.permute(0, 2, 3, 1).contiguous()
        x_theta = x_theta.view(n * h * w, 1, c)

        x_phi = F.unfold(x_phi, kernel_size=(kh, kw), stride=1, padding=pad)
        x_phi = x_phi.contiguous().view(n, c, kh * kw, h * w)
        x_phi = x_phi.permute(0, 3, 1, 2).contiguous()
        x_phi = x_phi.view(n * h * w, c, kh * kw)

        out = torch.matmul(x_theta, x_phi)
        out = out.view(n, h, w, kh * kw)

        return out

    @staticmethod
    def f_weighting(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad)
        x_theta = x_theta.permute(0, 2, 1).contiguous()
        x_theta = x_theta.view(n * h * w, c, kh * kw)

        x_phi = x_phi.view(n * h * w, kh * kw, 1)

        out = torch.matmul(x_theta, x_phi)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        weight = self.f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(x3, weight, self.kH, self.kW)

        return out

""" Convolutional Residual Units """
class LocalAttentionLayers(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW, n_layers):
        super().__init__()

        self.local_attention_blocks = nn.ModuleList([TorchLocalAttention(inp_channels, out_channels, kH, kW) for i in range(n_layers)])
    
    def forward(self, x):
        for i, layer in enumerate(self.local_attention_blocks):
            x = layer(x)
        return x

""" Convolutional Residual Units """
class ConvResLayers(nn.Module):
    def __init__(self, n_channels, exp_channels, n_layers):
        super().__init__()

        self.res_blocks = nn.ModuleList([ResBottleNeckBlock(n_channels, exp_channels) for i in range(n_layers)])
    
    def forward(self, x):
        for i, layer in enumerate(self.res_blocks):
            x = layer(x)
        return x

