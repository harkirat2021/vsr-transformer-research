import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py


""" Read video data from HDF5 file """


def read_hdf5(filepath, group_name):
    with h5py.File(filepath, "r") as f:
        # Get the data
        data = np.array(list(f[group_name]))

    return data


""" Write video data to HDF5 file """


def write_hdf5(data, filename, groupname):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("group_name", data=data)


""" Convert video into numpy """


def get_video_data(path, max_frames, img_size):
    if not os.path.exists(path):
        raise Exception('Path does not exist.')

    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    f = 0
    while ret and f < max_frames:
        f += 1
        ret, img = cap.read()

        try:
            img = cv2.resize(img, img_size)
        except Exception as e:
            ret = False

        if ret:
            frames.append(img)

    video = np.stack(frames, axis=0)
    video = np.moveaxis(video, -1, -3)
    video = video / 255

    return video


""" Divide data into sequences """


def prepare_sequences(data, seq_len):
    # Truncate sequence if not divisible by seq_len
    data = data[:seq_len * (data.shape[0] // seq_len)]

    # Split into sequences
    return np.array(np.split(data, data.shape[0] / seq_len))


""" Divide video frames into patches """


def prepare_patches(data, patch_shape, color_channel):
    # Safety check
    if color_channel:
        if len(data.shape) != 5:
            raise ValueError("Invalid data shape. Required shape: (t, s, c, h, w)")
        if data.shape[3] % patch_shape[0] != 0 or data.shape[4] % patch_shape[0] != 0:
            raise ValueError("Patch shape does evenly divide data")
        # Split patches
        data = np.array(np.split(data, data.shape[3] / patch_shape[0], axis=3))
        data = np.array(np.split(data, data.shape[5] / patch_shape[1], axis=5))

    else:
        if len(data.shape) != 4:
            raise ValueError("Invalid data shape. Required shape: (t, s, h, w)")
        if data.shape[2] % patch_shape[0] != 0 or data.shape[3] % patch_shape[0] != 0:
            raise ValueError("Patch shape does evenly divide data")

        # Split patches
        data = np.array(np.split(data, data.shape[2] / patch_shape[0], axis=2))
        #print(data.shape)
        data = np.array(np.split(data, data.shape[4] / patch_shape[1], axis=4))
        #print(data.shape)

    # Merge patche dimensions
    data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], *data.shape[3:]))
    #print(data.shape)

    return data


# NOT USED BUT STILL MIGHT MAYBE MIGHT KIND OF BE USEFUL... MAYBE

""" Split video data into batches of sequences """


def batchify(video, seq_len, batch_size):
    x = video[:-1]
    y = video[1:]

    # Shape into batches of batch size and sequence length
    x = x.reshape(-1, batch_size, seq_len, video.shape[3], video.shape[1], video.shape[2])
    y = y.reshape(-1, batch_size, seq_len, video.shape[3], video.shape[1], video.shape[2])

    # Need to make seq_len come before batch size
    x = np.swapaxes(x, 1, 2)
    y = np.swapaxes(y, 1, 2)

    return x, y

if __name__ == "__main__":

    #data = get_video_data("data/temp/leopard_climb.mp4", 1000, (640, 320))
    #data = np.flip(data, 1)
    #print(data.shape)

    print("patching")
    data = read_hdf5("data/temp/leopard_climb.hdf5", "")
    data = prepare_sequences(data, seq_len=5)

    print(data.shape)

    plt.imshow(np.swapaxes(np.swapaxes(data[1, 0, :, :, :], -3, -1), -2, -3))
    plt.show()

    data = prepare_patches(data, (320, 320))

    print(data.shape)

    plt.imshow(np.swapaxes(np.swapaxes(data[2, 0, :, :, :], -3, -1), -2, -3))
    plt.show()

    plt.imshow(np.swapaxes(np.swapaxes(data[2, 1, :, :, :], -3, -1), -2, -3))
    plt.show()

    plt.imshow(np.swapaxes(np.swapaxes(data[2, 2, :, :], -3, -1), -2, -3))
    plt.show()