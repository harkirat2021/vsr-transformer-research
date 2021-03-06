import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl

from src.model_modules import *

#### MODEL BASE ####

class VSRModelBase(pl.LightningModule):
    def __init__(self, name, scale, t, c, h, w):
        super(VSRModelBase, self).__init__()
        self.model_type = 'CNN'
        self.name = name
        self.scale = scale
        self.t = t
        self.h = h
        self.w = w
    
    ### Pytorch Lightning functions ###

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        # Compute loss between all, but first and last frame
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('valid_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer


#### VSR Models ####

class VSRTP1(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTP1, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        # TODO add more power in this layer
        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        # TODO - NEED TO CHANGE PATCH DIM
        self.patch_encoder = PatchEncode(t, c, h, w, embed_dim, 8)
        self.patch_decoder = PatchDecode(t, c, h, w, embed_dim, n_Convhidden, n_Convlayers)

        
        # Mask for now - sequence doesn't really make sense here does it?
        # TODO - NEED TO CHANGE PATCH DIM
        self.src_mask = self.generate_empty_mask(16).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
        # Forwad pass
        x = self.upsample(src)
        x = self.patch_encoder(x)
        x = torch.transpose(x, 0, 1)
        x = x * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = torch.transpose(x, 0, 1)
        y = self.patch_decoder(x)

        return y


class VSRTP2(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTP2, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(c, c, kernel_size=3, padding=1)

        # TODO - NEED TO CHANGE PATCH DIM
        self.patch_encoder = PatchEncode(t, c, h, w, embed_dim, 8)
        self.patch_decoder = PatchDecode(t, c, h, w, embed_dim, n_Convhidden, n_Convlayers)

        
        # Mask for now - sequence doesn't really make sense here does it?
        # TODO - NEED TO CHANGE PATCH DIM
        self.src_mask = self.generate_empty_mask(16).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
        # Forwad pass
        src = self.upsample(src)

        x = self.patch_encoder(src)
        x = torch.transpose(x, 0, 1)
        x = x * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = torch.transpose(x, 0, 1)
        x = self.patch_decoder(x)

        y = x + src # res connection

        return y

class VSRTP3(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTP3, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        self.convtrans1 = ConvTransform(c, n_Convhidden, n_Convlayers)
        self.convtrans2 = ConvTransform(c, n_Convhidden, n_Convlayers)

        #self.convtrans1 = VSRCNN1("", scale, t, c, h, w, n_Convhidden, n_Convlayers, use_upsample=False)
        #self.convtrans2 = VSRCNN1("", scale, t, c, h, w, n_Convhidden, n_Convlayers, use_upsample=False)

        # TODO add more power in this layer
        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        # TODO - NEED TO CHANGE PATCH DIM
        self.patch_encoder = PatchEncode(t, c, h, w, embed_dim, 8)
        self.patch_decoder = PatchDecode(t, c, h, w, embed_dim, n_Convhidden, n_Convlayers)

        # Mask for now - sequence doesn't really make sense here does it?
        # TODO - NEED TO CHANGE PATCH DIM
        self.src_mask = self.generate_empty_mask(16).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
        # Forwad pass
        x = self.upsample(src)
        x = self.convtrans1(x)
        x = self.patch_encoder(x)
        x = torch.transpose(x, 0, 1)
        x = x * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = torch.transpose(x, 0, 1)
        x = self.patch_decoder(x)
        y = self.convtrans2(x)

        return y

class VSRTP4(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTP4, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(c, c, kernel_size=3, padding=1)

        # Parallel tower
        self.convtrans = ConvTransform(c, n_Convhidden, n_Convlayers)

        # TODO - NEED TO CHANGE PATCH DIM
        self.patch_encoder = PatchEncode(t, c, h, w, embed_dim, 8)
        self.patch_decoder = PatchDecode(t, c, h, w, embed_dim, n_Convhidden, n_Convlayers)

        
        # Mask for now - sequence doesn't really make sense here does it?
        # TODO - NEED TO CHANGE PATCH DIM
        self.src_mask = self.generate_empty_mask(16).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
        # Forwad pass
        src = self.upsample(src)
        
        cty = self.convtrans(src)

        x = self.patch_encoder(src)
        x = torch.transpose(x, 0, 1)
        x = x * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = torch.transpose(x, 0, 1)
        x = self.patch_decoder(x)

        y = x + src + cty # triple res connection

        return y

# VSRTP5 - with variable patch size

class VSRTE1(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTE1, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'
        self.name = name

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        # Scale input and output shape down by scale since we post upsample
        self.frame_encoder = FrameSeqEncoder(c, h // scale, w // scale, embed_dim, n_Convhidden, n_Convlayers, n_stride)
        self.frame_decoder = FrameSeqDecoder(c, h // scale, w // scale, embed_dim, n_Convhidden, n_Convlayers, n_stride)

        # TODO add more power in this layer
        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        # Mask for now - sequence doesn't really make sense here does it?
        self.src_mask = self.generate_empty_mask(t).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
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

""" VSRTE2 """
class VSRTE2(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_heads, n_transformerhidden, n_transformerlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRTE2, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'Transformer'

        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, n_heads, n_transformerhidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformerlayers)

        self.frame_encoder = FrameSeqEncoder(c, h, w, embed_dim, n_Convhidden, n_Convlayers, n_stride)
        self.frame_decoder = FrameSeqDecoder(c, h, w, embed_dim, n_Convhidden, n_Convlayers, n_stride)

        # TODO add more power in this layer
        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

        # Mask for now - sequence doesn't really make sense here does it?
        self.src_mask = self.generate_empty_mask(t).to(self.device) # Move to model device

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
      
    def generate_empty_mask(self, sz):
        mask = torch.zeros((sz, sz)).transpose(0, 1)
        return mask

    def forward(self, src):
        # Move to model device if not on
        if self.device != self.src_mask.device:
            self.src_mask = self.src_mask.to(self.device)
            
        # Swap batch and sequence dimension
        src = torch.transpose(src, 0, 1)

        # Forwad pass
        x = self.upsample(src)
        x = self.frame_encoder(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        y = self.frame_decoder(x)
        

        # Swap back batch and sequence dimension
        y = torch.transpose(y, 0, 1)

        return y

""" VSR Seq Autoencoder """
class VSRSA1(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, embed_dim, n_esthidden, n_estlayers, n_Convhidden, n_Convlayers, n_stride, dropout):
        super(VSRSA1, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'SequenceAutoencoder'

        self.frame_encoder = FrameSeqEncoder(c, h // scale, w // scale, embed_dim, n_Convhidden, n_Convlayers, n_stride)
        self.frame_decoder = FrameSeqDecoder(c, h // scale, w // scale, embed_dim, n_Convhidden, n_Convlayers, n_stride)
        self.embedding_seq_transform = EmbeddingSeqTransform(t, embed_dim, n_esthidden, n_estlayers)
        
        # TODO add more power in this layer
        self.upsample = UpsampleSeqLayer(seq_len=t, n_features=c, k=3, factor=scale)

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

""" VSR CNN """
class VSRCNN1(VSRModelBase):
    def __init__(self, name, scale, t, c, h, w, n_hidden, n_layers, use_upsample=True):
        super(VSRCNN1, self).__init__(name, scale, t, c, h, w)
        self.model_type = 'CNN'
        self.use_upsample = use_upsample

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=n_hidden//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_hidden//2, out_channels=n_hidden, kernel_size=3, stride=1, padding=1)
        self.layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(n_hidden, n_hidden, 3, stride=1, padding=1)
                ) for i in range(n_layers)
            ]
        )
        self.conv3 = nn.Conv2d(in_channels=n_hidden, out_channels=c, kernel_size=3, stride=1, padding=1)

        self.up_sample = nn.Upsample(scale_factor=2)

    """ (batch_dim, time_dim, channel_dim, height_dim, width_dim) -> (batch_dim, time_dim, channel_dim, height_dim, width_dim) """
    def forward(self, x):
        # Swap batch and sequence dimension
        x = torch.transpose(x, 0, 1)

        y_seq = []
        for xi in x:
            if self.use_upsample:
                xi = F.relu(self.conv2(self.up_sample(F.relu(self.conv1(self.up_sample(xi))))))
            else:
                xi = F.relu(self.conv2(F.relu(self.conv1(xi))))
            for layer in self.layers:
                xi = F.relu(layer(xi))
            y_seq.append(F.sigmoid(self.conv3(xi)))
          
        y = torch.stack(y_seq, dim=0)

        # Swap back batch and sequence dimension
        y = torch.transpose(y, 0, 1)

        return y

""" UNet using Residual BottleNeck Blocks and custom transform unit """
class VSRRU_1(nn.Module):
    def __init__(self, transform_unit, n_channels, exp_channels, bottle_neck_exp_channels, n_layers, scale_factor):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode='trilinear')

        self.transform_unit = transform_unit

        self.expand_cov = nn.Conv2d(n_channels, exp_channels, kernel_size=3, padding=1)
        self.compress_cov = nn.Conv2d(exp_channels, n_channels, kernel_size=3, padding=1)

        self.res_blocks_in = nn.ModuleList([ResBottleNeckBlock(n_channels, bottle_neck_exp_channels) for i in range(n_layers)])
        self.res_blocks_out = nn.ModuleList([ResBottleNeckBlock(n_channels, bottle_neck_exp_channels) for i in range(n_layers)])
    
    def forward(self, x):
        # Upsample
        x = self.up(x)

        # Join frames with batch
        in_shape = x.shape
        x = x.reshape((in_shape[0] * in_shape[1], in_shape[2], in_shape[3], in_shape[4]))

        # In res blocks
        x_i = []
        for i, layer in enumerate(self.res_blocks_in):
            x = layer(x)
            x_i.append(x.clone())

        # Expand to exand channels
        x = F.relu(self.expand_cov(x))

        # Transform unit
        x = self.transform_unit(x)
        
        # Compress from exand channels
        x = F.relu(self.compress_cov(x))

        # Out res blocks
        x_i.reverse()
        for i, layer in enumerate(self.res_blocks_out):
            x = layer(x + x_i[i])
        
        # Split frames and batch
        x = x.reshape(in_shape)

        return x

class VSRRU_LA_1(pl.LightningModule):
    def __init__(self, n_transform_layers, n_channels, exp_channels, bottle_neck_exp_channels, n_layers, scale_factor):
        super().__init__()
        self.transform_unit = LocalAttentionLayers(exp_channels, exp_channels, kH=7, kW=7, n_layers=n_transform_layers)
        self.model = VSRRU_1(self.transform_unit, n_channels, exp_channels, bottle_neck_exp_channels, n_layers, scale_factor)
    
    def forward(self, x):
        return self.model(x)

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        # Compute loss between all, but first and last frame
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('valid_loss', loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer

 # changes
class VSRRU_CV_1(pl.LightningModule):
    def __init__(self, n_transform_layers, n_channels, exp_channels, bottle_neck_exp_channels, n_layers, scale_factor):
        super().__init__()
        self.transform_unit = ConvResLayers(exp_channels, exp_channels, n_transform_layers)
        self.model = VSRRU_1(self.transform_unit, n_channels, exp_channels, bottle_neck_exp_channels, n_layers, scale_factor)
        
    def forward(self, x):
        return self.model(x)

    def mse_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        # Compute loss between all, but first and last frame
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        print(x.shape)
        outputs = self.forward(x)
        loss = self.mse_loss(outputs[:,outputs.shape[1]//2+1,:,:,:], y[:,0,:,:,:])
        self.log('valid_loss', loss)
        print(loss)

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
      return optimizer