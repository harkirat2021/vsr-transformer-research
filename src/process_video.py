import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    # Scale down to 0..1 range
    return video / 255

""" Divide video frames into patches """
# TODO