import numpy as np
import argparse
import yaml

from src.train import *
from src.evaluate import *
from src.process_video import *
from src.data import *

parser = argparse.ArgumentParser(description="Run the deep Q trading agent")
parser.add_argument('--task', type=str, help="'train' or 'evaluate'")

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

if __name__ == "__main__":
    print("Loading video...")
    # TEMP
    video = get_video_data("data/temp/the_muffin_man.mp4", 400, (64, 36))
    data_module = VideoDataModule(video, config["SEQ_LEN"])
    train(data_module=data_module, seq_len=config["SEQ_LEN"], gpus=0)

    """
    if args.task == "train":
        train(data_module=data_module, seq_len=config["SEQ_LEN"], gpus=0)

    elif args.task == "evaluate":
        evaluate(data_module=data_module, gpus=0)
    """
