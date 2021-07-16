import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_video_data(path, max_frames):
    if not os.path.exists(path):
        raise Exception('Path does not exist.')

    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    f = 0
    while ret and f < max_frames:
        f += 1
        ret, img = cap.read() 
        if ret:
            frames.append(img)

    video = np.stack(frames, axis=0)
    return video

video = get_video_data("data/temp/the_muffin_man.mp4", 200)

def batchify(video):
    