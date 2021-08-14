import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from src.train import *
from src.evaluate import *
from src.process_data import *
from src.data import *
from src.metrics import *
import h5py

""" Read video data from HDF5 file """
def read_hdf5(filepath, group_name):
    with h5py.File(filepath, "r") as f:
        # Get first group
        if not group_name:
            group_name = list(f.keys())[0]

        # Get the data
        data = np.array(list(f[group_name]))
    
    return data

data = read_hdf5("data/temp/test_data.hdf5", "data_hr")
plt.imshow(data[43][0])
plt.show()

plt.imshow(data[43][0])
plt.show()

plt.imshow(data[43][0])
plt.show()

print(poop)
plt.imshow(np.swapaxes(np.swapaxes(model(x[6002:6003])[0][0][:][:][:].detach().numpy(), -3, -1), -2, -3))
plt.show()

plt.imshow(np.swapaxes(np.swapaxes(y[6002][0][:][:][:], -3, -1), -2, -3))
plt.show()
print("Done")