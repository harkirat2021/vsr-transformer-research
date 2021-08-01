import numpy as np
import torch
import cv2
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader

from src.process_data import *

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path, valid_data_path, seq_len, patch_shape):
        super(VideoDataModule, self).__init__()

        self.train_dataset = self.prepare_dataset(train_data_path, seq_len, patch_shape)
        self.valid_dataset = self.prepare_dataset(valid_data_path, seq_len, patch_shape)

    def prepare_dataset(self, data_path, seq_len, patch_shape):
        data = read_hdf5(filepath=data_path, group_name="")

        # Downsample - TODO wont need for zeus
        mp = torch.nn.MaxPool3d((1,2,2), stride=(1,2,2))

        # Set inputs and outputs
        x = data.copy()
        y = data.copy()

        # Split x into sequences and convert to tensor
        x = prepare_sequences(x, seq_len=seq_len)
        x = prepare_patches(x, patch_shape=patch_shape)
        x = torch.tensor(x).float()
        x = mp(x)

        # Split y into sequences and convert to tensor
        y = prepare_sequences(y, seq_len=seq_len)
        y = prepare_patches(y, patch_shape=patch_shape)
        y = torch.tensor(y).float()

        # Set dataset
        dataset = TensorDataset(x, y)

        return dataset

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=4)