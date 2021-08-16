import numpy as np
import torch
import cv2
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader
from src.process_data import *
from sklearn.model_selection import train_test_split

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, train_data_path, batch_size, scale, seq_len, patch_shape, train_valid_split, has_color_channel,
                 prepared_seq=False, prepared_patch=False):
        super(VideoDataModule, self).__init__()

        self.dataset_name = dataset_name
        self.scale = scale
        self.batch_size = batch_size

        self.train_dataset, self.valid_dataset = self.prepare_dataset(train_data_path, seq_len, patch_shape, train_valid_split,
                                                                      has_color_channel, prepared_seq, prepared_patch)

    # Crop patches to be a power of 2
    def center_crop_data(self, data):
        original_w = data.shape[-1]
        resize_w = 2**int(np.log(original_w)/np.log(2))
        original_h = data.shape[-2]
        resize_h = 2**int(np.log(original_h)/np.log(2))

        edge_reduce_h = (original_h-resize_h) // 2
        edge_reduce_w = (original_w-resize_w) // 2

        return data[:,:,:,edge_reduce_h:original_h-edge_reduce_h,edge_reduce_w:original_w-edge_reduce_w]

    def prepare_dataset(self, data_path, seq_len, patch_shape, train_valid_split, has_color_channel, prepared_seq, prepared_patch):
        y = read_hdf5(filepath=data_path, group_name="data_hr")
        x = read_hdf5(filepath=data_path, group_name="data_lr")

        # Add 1D color channel if data does not have it
        if not has_color_channel:
            x = np.expand_dims(x, axis=-3)
            y = np.expand_dims(y, axis=-3)
            # Now has color channel
            has_color_channel = True

        # Split x, y into sequences and convert to tensor
        if not prepared_seq:
            x = prepare_sequences(x, seq_len)
            y = prepare_sequences(y, seq_len)
        if not prepared_patch:
            x = prepare_patches(x, patch_shape, has_color_channel)
            y = prepare_patches(y, patch_shape, has_color_channel)
        
        # Adjust size of patch to be power of 2
        # TODO - GET RID OF THIS WHEN MAKING CUSTOM DATA
        x = self.center_crop_data(x)
        y = self.center_crop_data(y)

        # Convert to torch tensors
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        # Downsample inputs
        mp = torch.nn.MaxPool3d((1, self.scale, self.scale), stride=(1, self.scale, self.scale))
        x = mp(x)

        # Split to train, valid set
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=train_valid_split, shuffle=False)

        # Set dataset
        trainset = TensorDataset(x_train, y_train)
        validset = TensorDataset(x_valid, y_valid)

        return trainset, validset

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)