import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_data import *

if __name__ == "__main__":
    rootdir = '/home/harkirat/data/vimeo/vimeo_septuplet/sequences'
    save_dir = "/home/harkirat/data/vimeo/processed"

    print("Reading files...")
    seqs = []
    for subdir, dirs, files in os.walk(rootdir):
        frames = []
        for file in files:
            im = cv2.imread(os.path.join(subdir, file)) / 255
            frames.append(im)
        if len(frames) > 0:
            frames_arr = np.stack(frames, axis=0)
            seqs.append(frames_arr)

            if len(seqs) % 100 == 0:
                print("{}/91701...".format(len(seqs)))
        if len(seqs) == 100:
            break

    print("Processing...")
    hr = np.stack(seqs, axis=0)

    hr = np.swapaxes(hr,-1,-3)
    hr = np.swapaxes(hr,-1,-2)
    hr = prepare_patches(hr, patch_shape=(32, 32), color_channel=True)

    hr_train = hr[:int(0.8*hr.shape[0])]
    hr_test = hr[int(0.8*hr.shape[0]):]

    print("train: ", hr_train.shape)
    print("test: ", hr_test.shape)

    print("Saving...")
    
    with h5py.File(os.path.join(save_dir, "vimeo_train_sample.hdf5"), "w") as data_file:
        data_file.create_dataset("data_hr", data=hr_train)
        data_file.create_dataset("data_lr", data=hr_train)

    with h5py.File(os.path.join(save_dir, "vimeo_test_sample.hdf5"), "w") as data_file:
        data_file.create_dataset("data_hr", data=hr_test)
        data_file.create_dataset("data_lr", data=hr_test)
    
    print("Done.")
