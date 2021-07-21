import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

class VideoDataModule(pl.LightningDataModule):
  def __init__(self, data, seq_len):
      super(VideoDataModule, self).__init__()

      # Split into sequences and convert to tensor
      data = np.split(data, data.shape[0] / seq_len)
      data = torch.tensor(data).float()

      self.x = data.clone()
      self.y = data.clone()

  def setup(self, stage):
      pass

  def train_dataloader(self):
      return DataLoader(TensorDataset(self.x, self.y), batch_size=4)

  def val_dataloader(self):
      return DataLoader(TensorDataset(self.x, self.y), batch_size=4)