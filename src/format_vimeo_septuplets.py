import os
import sys
import cv2
import numpy as np
import h5py
from process_data import *

# Create numpy array of eptuplets from given folders
SEPTUPLET_DIR = "data/temp/temp_sep_data"
GROUP_DIRS = ["00001"]
OUTPUT_DIR = "data/temp/temp_processed"

# Loop through each image
def run_format_sep(sep_dir, group_dirs, out_dir, data_name):
    seps = []
    for group_name in group_dirs:
        for sep_path in os.listdir(os.path.join(sep_dir, group_name)):
            sep = []
            for filename in os.listdir(os.path.join(sep_dir, group_name, sep_path)):
                img = cv2.imread(os.path.join(sep_dir, group_name, sep_path, filename))
                if img is not None:
                    sep.append(img)

            if len(sep) == 7:
                sep_arr = np.stack(sep)
                seps.append(sep_arr)
            else:
                print("no sep: ", len(sep))

    septuplet_data = np.stack(seps)

    septuplet_data = np.swapaxes(septuplet_data,-1,-3)
    septuplet_data = np.swapaxes(septuplet_data,-1,-2)

    print(septuplet_data.shape)
    septuplet_patches = prepare_patches(septuplet_data, patch_shape=(32, 32), color_channel=True)
    print(septuplet_patches.shape)

    # Save
    with h5py.File(out_dir, 'a') as hf:
        hf.create_dataset("data_lr",  data=septuplet_patches)
        hf.create_dataset("data_hr",  data=septuplet_patches)

if __name__ == "__main__":
    sep_dir = sys.argv[1]
    group_dirs = sys.argv[2].split(",")
    out_dir = sys.argv[3]
    data_name = sys.argv[3]
    run_format_sep(sep_dir, group_dirs, out_dir, data_name)
        