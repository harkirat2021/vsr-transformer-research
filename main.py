import numpy as np
import argparse
import yaml

from src.train import *
from src.evaluate import *
from src.process_data import *
from src.data import *

parser = argparse.ArgumentParser(description="Run the deep Q trading agent")
parser.add_argument('--task', type=str, help="'train' or 'evaluate'")
parser.add_argument('--n_Convhidden', type=int, default=12, help='number of CNN hidden units')
parser.add_argument('--n_Convlayers', type=int, default=0, help='number of CNN hidden layers')
parser.add_argument('--n_stride', type=int, default=2, help='value of stride')

args = parser.parse_args()

# Get all config values
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

if __name__ == "__main__":
    # Init data
    print("Loading data...")
    data_module = VideoDataModule(train_data_path="data/temp/the_muffin_man.hdf5", valid_data_path="data/temp/the_muffin_man.hdf5", seq_len=config["SEQ_LEN"], patch_shape=config["PATCH_SHAPE"])

    # Init model - TODO option to load from checkpoint
    print("Initializing model...")
    model = VSRTE1(name="sample_model", c=3, h=8, w=8, embed_dim=8, n_head=4, h_dim=20, n_layers=2, dropout=0.5, n_Convhidden =args.n_Convhidden, n_Convlayers=args.n_Convlayers, n_stride=args.n_stride)
    model.set_src_mask(config["SEQ_LEN"])
    
    if args.task == "train":
        print("Training...")
        model = train(model=model, data_module=data_module, max_epochs=2, gpus=0, save=False)

    elif args.task == "eval":
        # TODO - save to file
        print("Evaluating...")
        psnr, ssim = evaluate(model=model, data_module=data_module)
        print("PSNR: {} SSIM: {}".format(psnr, ssim))
    
    print("Done")
