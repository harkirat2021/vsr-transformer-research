import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from src.train import *
from src.evaluate import *
from src.process_data import *
from src.data import *

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Get all config values
with open("model_settings.yml", "r") as ymlfile:
    model_settings = yaml.load(ymlfile, Loader=yaml.FullLoader)

# Init data
print("Loading data...")
data_module = VideoDataModule("bob", train_data_path=config["DATA"]["SAMPLE_NATGEO"]["TRAIN"], valid_data_path=config["DATA"]["SAMPLE_NATGEO"]["VALID"], seq_len=config["SEQ_LEN"], patch_shape=config["PATCH_SHAPE"])
x = data_module.train_dataset[:][0]
y = data_module.train_dataset[:][1]
print(x.shape)
print(y.shape)

plt.imshow(np.swapaxes(np.swapaxes(x[6002][0][:][:][:], -3, -1), -2, -3))
plt.show()


# Init model - TODO option to load from checkpoint
print("Initializing model...")
model = VSRTE1(name="bob", scale=config["SCALE"], t=config["SEQ_LEN"], c=3, h=16, w=16, **model_settings["VSRTE1_SAMPLE"])

plt.imshow(np.swapaxes(np.swapaxes(model(x[6002:6003])[0][0][:][:][:].detach().numpy(), -3, -1), -2, -3))
plt.show()

plt.imshow(np.swapaxes(np.swapaxes(y[6002][0][:][:][:], -3, -1), -2, -3))
plt.show()
print("Done")