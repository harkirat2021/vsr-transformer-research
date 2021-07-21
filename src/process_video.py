import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

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
    
def read_hdf5(filename, group_name):
    with h5py.File(filename, "r") as f:
        # TODO - only gets group 1
        # List all groups
        a_group_key = list(f.keys())[0]

        # Get the data
        data = np.array(list(f[a_group_key]))
    
    return data

def write_hdf5(data, filename, groupname):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("group_name", data=data)

""" Divide video frames into patches """
# TODO

if __name__ == "__main__":
    #v = read_hdf5("data/temp/the_muffin_man.hdf5", "temp")
    #print(v.shape)
    #assert False
    v = get_video_data("data/temp/the_muffin_man.mp4", 201, (640, 360))
    print(v.shape)
    write_hdf5(v, "data/temp/the_muffin_man.hdf5", "temp")