import numpy as np
import argparse
import yaml

from src.train import *
from src.evaluate import *
from src.process_data import *
from src.data import *

parser = argparse.ArgumentParser(description="Run the VSR pipeline")
parser.add_argument('--task', type=str, help="'train' or 'evaluate'", required=True)
parser.add_argument('--model_type', type=str, help="model architecture", required=True)
parser.add_argument('--model_settings', type=str, help="params data to initialize model", required=True)
parser.add_argument('--model_path', type=str, help="path of model")
parser.add_argument('--data', type=str, help="dataset to use for train, valid, and test", required=True)
parser.add_argument('--num_epochs', type=int, help="number of epochs to train for")
parser.add_argument('--load_checkpoint', type=str, default=False, help="load weights, biases and hyperparameters from checkpoint")
parser.add_argument('--model_save', type=str, help="flag for whether or not to save the model")

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

    # Set experiment directory
    experiment_dir = os.path.join(args.data.lower(), args.model_settings.lower(), "scale_{}".format(config["SCALE"]), "patch_{}x{}x{}".format(config["NUM_CHANNELS"], *config["HR_PATCH_SHAPE"]))

    # Init data
    print("Loading data...")
    data_module = VideoDataModule(dataset_name=args.data.lower(),train_data_path=config["DATA"][args.data.upper()]["TRAIN"], valid_data_path=config["DATA"][args.data.upper()]["VALID"], scale=config["SCALE"], seq_len=config["SEQ_LEN"], patch_shape=config["HR_PATCH_SHAPE"])
    print(data_module.train_dataset[:][0].shape)
    print(data_module.train_dataset[:][1].shape)

    # Init model - TODO option to load from checkpoint
    print("Initializing model...")
    if str(args.model_type).lower() == "vsrsa1":
        if args.load_checkpoint == False:
            model = VSRSA1(name=args.model_settings.lower(),
                            scale=config["SCALE"], t=config["SEQ_LEN"], c=config["NUM_CHANNELS"],
                            h=config["HR_PATCH_SHAPE"][0] // config["SCALE"], w=config["HR_PATCH_SHAPE"][1] // config["SCALE"],
                            **model_settings[args.model_settings.upper()])
        else:
            model = VSRSA1.load_from_checkpoint(checkpoint_path="experiment/VSRSA1/version_x/")
    
    elif str(args.model_type).lower() == "vsrte1":
        if args.load_checkpoint == False:
            model = VSRTE1(name=args.model_settings.lower(),
                            scale=config["SCALE"], t=config["SEQ_LEN"], c=config["NUM_CHANNELS"],
                            h=config["HR_PATCH_SHAPE"][0] // config["SCALE"], w=config["HR_PATCH_SHAPE"][1] // config["SCALE"],
                            **model_settings[args.model_settings.upper()])
        else:
            model = VSRTE1.load_from_checkpoint(checkpoint_path="experiment/VSRTE1/version_x/")
    
    # Run task
    if args.task == "train":
        print("Training...")
        model = train(model=model, data_module=data_module, experiment_dir=experiment_dir, max_epochs=args.num_epochs, gpus=0, save=bool(args.model_save))
    elif args.task == "eval":
        print("Evaluating...")
        psnr, ssim = evaluate(model=model, data_module=data_module)
        print("PSNR: {} SSIM: {}".format(psnr, ssim))
        # TODO - save to file
    elif args.task == "sample":
        print("Sampling...")
        #sample(model=model, data_module=data_module)

    print("Done")

#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