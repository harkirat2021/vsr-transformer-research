import numpy as np
import argparse
import yaml

from src.train import *
from src.evaluate import *
from src.process_data import *
from src.data import *

parser = argparse.ArgumentParser(description="Run the deep Q trading agent")
parser.add_argument('--task', type=str, help="'train' or 'evaluate'", required=True)
parser.add_argument('--model_path', type=str, help="path of model")
parser.add_argument('--model_type', type=str, help="model architecture", required=True)
parser.add_argument('--model_settings', type=str, help="params data to initialize model", required=True)
parser.add_argument('--load_checkpoint', type=str, default=False, help="load weights, biases and hyperparameters from checkpoint")


args = parser.parse_args()

if __name__ == "__main__":
    # Set model save
    args.model_save = args.model_save == "True"

    # Get all config values
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Get all config values
    with open("model_settings.yml", "r") as ymlfile:
        model_settings = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    # Init data
    print("Loading data...")
    data_module = VideoDataModule(train_data_path="data/temp/the_muffin_man.hdf5", valid_data_path="data/temp/the_muffin_man.hdf5", seq_len=config["SEQ_LEN"], patch_shape=config["PATCH_SHAPE"])

    # Init model - TODO option to load from checkpoint
    print("Initializing model...")
    if str(args.model_type).lower() == "vsrsa1":
        if args.load_checkpoint == False:
            model = VSRSA1(name="sample_model", scale=config["SCALE"], t=5, c=3, h=8, w=8, **model_settings[args.model_settings.upper()])
        else:
            model = VSRSA1.load_from_checkpoint(checkpoint_path="experiment/VSRSA1/version_x/")
    elif str(args.model_type).lower() == "vsrte1":
        if args.load_checkpoint == False:
            model = VSRTE1(name="sample_model", scale=config["SCALE"], t=5, c=3, h=8, w=8, **model_settings[args.model_settings.upper()])
        else:
            model = VSRTE1.load_from_checkpoint(checkpoint_path="experiment/VSRTE1/version_x/")

    
    # Run task
    if args.task == "train":
        print("Training...")
        model = train(model=model, data_module=data_module, max_epochs=2, gpus=0, save=bool(args.model_save))

    elif args.task == "eval":
        print("Evaluating...")
        psnr, ssim = evaluate(model=model, data_module=data_module)
        print("PSNR: {} SSIM: {}".format(psnr, ssim))
        # TODO - save to file
    
    print("Done")

#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