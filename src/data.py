import numpy as np
import torch
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

        # Split into sequences and convert to tensor
        data = prepare_sequences(data, seq_len=seq_len)
        data = prepare_patches(data, patch_shape=patch_shape)
        data = torch.tensor(data).float()

        data = data[:1000]

        # Set inputs and outputs
        x = data.clone()
        y = data.clone()

        # Set dataset
        dataset = TensorDataset(x, y)

        return dataset

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=4)