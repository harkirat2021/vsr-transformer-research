import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

""" Read video data from HDF5 file """
def read_hdf5(filepath, group_name):
    with h5py.File(filepath, "r") as f:
        # TODO - only gets group 1
        # List all groups
        a_group_key = list(f.keys())[0]

        # Get the data
        data = np.array(list(f[a_group_key]))
    
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
        img = cv2.resize(img, img_size)
        if ret:
            frames.append(img)

    video = np.stack(frames, axis=0)
    video = np.moveaxis(video,  -1, -3)
    video = video / 255
    
    return video

""" Divide data into sequences """
def prepare_sequences(data, seq_len):
    # Truncate sequence if not divisible by seq_len
    data = data[:data.shape[0] // seq_len]

    # Split into sequences
    return np.array(np.split(data, data.shape[0] / seq_len))

# TODO - DOESNT WORK AT ALL
""" Divide video frames into patches """
def prepare_patches(data, patch_shape):
    # TODO - shape safety check

    # Split patches
    print("o", data[0,0, 0, :8, :8])
    data = np.array(np.split(data, data.shape[3] / patch_shape[0], axis=3))
    print("h", data[0, 0, 0, 0, :, :8])
    data = np.array(np.split(data, data.shape[5] / patch_shape[1], axis=5))
    print("v", data[0, 0, 0, 0, 0, :, :])

    # Merge patche dimensions
    data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], *data.shape[3:]))

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
    print("patching")
    data = read_hdf5("data/temp/the_muffin_man.hdf5", "")
    data = prepare_sequences(data, seq_len=5)

    plt.imshow(data[0, 0, 0, :8, :8])
    plt.show()

    data = prepare_patches(data, (8, 8))
    print(data.shape)

    plt.imshow(data[0, 0, 0, :, :])
    plt.show()

    assert False

    print(data.shape)
    plt.imshow(np.swapaxes(data[0][0], -3, -1))
    print(np.max(data[0][0]), np.mean(data[0][0]), np.min(data[0][0]))
    plt.show()

    data = prepare_patches(data, (8, 8))
    print(data.shape)
    plt.imshow(np.swapaxes(data[0][0][0][0], -3, -1))
    print(np.max(data[0][0]), np.mean(data[0][0]), np.min(data[0][0]))
    plt.show()

    assert False

    #v = read_hdf5("data/temp/the_muffin_man.hdf5", "temp")
    #print(v.shape)
    #assert False
    v = get_video_data("data/temp/the_muffin_man.mp4", 201, (640, 360))
    print(v.shape)
    write_hdf5(v, "data/temp/the_muffin_man.hdf5", "temp")