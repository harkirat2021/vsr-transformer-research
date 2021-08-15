import numpy as np
import argparse
import yaml

from src.train import *
from src.evaluate import *
from src.sample_outputs import *
from src.process_data import *
from src.data import *

# experiments/sample_natgeo/vsrsa1_sample/scale_4/patch_3x32x32/version_1/checkpoints/epoch=1-step=4099.ckpt
# python3 main.py --task train --model_type vsrsa1 --model_settings vsrsa1_sample --data sample_natgeo --num_epochs 2  --model_save True --check_load_path experiments/sample_natgeo/vsrsa1_sample/scale_4/patch_3x32x32/version_1/checkpoints/epoch=1-step=4099.ckpt
# python3 main.py --task eval --model_type vsrsa1 --model_settings vsrsa1_sample --data sample_natgeo --check_load_path experiments/sample_natgeo/vsrsa1_sample/scale_4/patch_3x32x32/version_0/checkpoints/epoch=4-step=10249.ckpt
# python3 main.py --task train --model_type vsrte1 --model_settings vsrte1_sample --data sample_natgeo --num_epochs 5  --model_save True

parser = argparse.ArgumentParser(description="Run the VSR pipeline")
parser.add_argument('--task', type=str, help="'train' or 'evaluate'", required=True)
parser.add_argument('--model_type', type=str, help="model architecture", required=True)
parser.add_argument('--model_settings', type=str, help="params data to initialize model", required=True)

parser.add_argument('--data', type=str, help="dataset to use for train, valid, and test", required=True)

parser.add_argument('--num_epochs', type=int, help="number of epochs to train for")
parser.add_argument('--num_gpus', type=int, default=0, help="number of gpus to train (0 to use cpu)")

parser.add_argument('--model_load', type=str, help="flag for whether or not to load the model")
parser.add_argument('--model_save', type=str, help="flag for whether or not to save the model")

args = parser.parse_args()

if __name__ == "__main__":
    # Set model save and load as boolean
    args.model_load = args.model_load == "True"
    args.model_save = args.model_save == "True"

    # Get all config values
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Get all model settings
    with open("model_settings.yml", "r") as ymlfile:
        model_settings = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Set experiment directory
    experiment_dir = os.path.join(config["EXPERIMENT_SAVE_DIR"], args.data.lower(), args.model_settings.lower(), "scale_{}".format(config[args.data.upper()]["SCALE"]), "patch_{}x{}x{}".format(config[args.data.upper()]["NUM_CHANNELS"], *config[args.data.upper()]["HR_PATCH_SHAPE"]))
    print(experiment_dir)

    # Set checkpoint load path
    # TODO - may be issue if multiple checkpoints, but not big concern for now
    check_load_path = ""
    if args.model_load:
        checkpoint_paths = []
        for subdir, dirs, files in os.walk(experiment_dir):
            for f in files:
                if "epoch=" in f:
                    path = os.path.join(subdir, f)
                    checkpoint_paths.append((path, path[path.index("version_") + len("version_")]))
        
        # Choose checkpoint from the latest version to load if paths list is not empty
        if checkpoint_paths:
            check_load_path = max(checkpoint_paths, key=lambda x: x[1])[0]

    # Init data
    print("Loading data...")
    data_module = VideoDataModule(dataset_name=args.data.lower(), train_data_path=config[args.data.upper()]["PATH"],
                                batch_size=config["BATCH_SIZE"], scale=config[args.data.upper()]["SCALE"],
                                seq_len=config[args.data.upper()]["SEQ_LEN"], patch_shape=config[args.data.upper()]["HR_PATCH_SHAPE"],
                                train_valid_split=config["TRAIN_VALID_SPLIT"], has_color_channel=False,
                                prepared_seq=True, prepared_patch=True)
    print("LR data shape: ", data_module.train_dataset[:][0].shape)
    print("HR data shape: ", data_module.train_dataset[:][1].shape)

    #### Init model ####
    
    # VSRSA1
    if str(args.model_type).lower() == "vsrsa1":
        print("Initializing model...")
        model = VSRSA1(name=args.model_settings.lower(),
                        scale=config[args.data.upper()]["SCALE"], t=config[args.data.upper()]["SEQ_LEN"], c=config[args.data.upper()]["NUM_CHANNELS"],
                        h=config[args.data.upper()]["HR_PATCH_SHAPE"][0] // config[args.data.upper()]["SCALE"],
                        w=config[args.data.upper()]["HR_PATCH_SHAPE"][1] // config[args.data.upper()]["SCALE"],
                        **model_settings[args.model_settings.upper()])
        # Load model from checkpoint
        if check_load_path:
            print("Loading model from checkpoint...")
            model = model.load_from_checkpoint(checkpoint_path=check_load_path,
                            name=args.model_settings.lower(),
                            scale=config[args.data.upper()]["SCALE"], t=config[args.data.upper()]["SEQ_LEN"], c=config[args.data.upper()]["NUM_CHANNELS"],
                            h=config[args.data.upper()]["HR_PATCH_SHAPE"][0] // config[args.data.upper()]["SCALE"],
                            w=config[args.data.upper()]["HR_PATCH_SHAPE"][1] // config[args.data.upper()]["SCALE"],
                            **model_settings[args.model_settings.upper()])
    
    # VSRTE1
    elif str(args.model_type).lower() == "vsrte1":
        print("Initializing model...")
        model = VSRTE1(name=args.model_settings.lower(),
                        scale=config[args.data.upper()]["SCALE"], t=config[args.data.upper()]["SEQ_LEN"], c=config[args.data.upper()]["NUM_CHANNELS"],
                        h=config[args.data.upper()]["HR_PATCH_SHAPE"][0] // config[args.data.upper()]["SCALE"],
                        w=config[args.data.upper()]["HR_PATCH_SHAPE"][1] // config[args.data.upper()]["SCALE"],
                        **model_settings[args.model_settings.upper()])
        # Load model from checkpoint
        if check_load_path:
            print("Loading model from checkpoint...")
            model = model.load_from_checkpoint(checkpoint_path=check_load_path,
                            name=args.model_settings.lower(),
                            scale=config[args.data.upper()]["SCALE"], t=config[args.data.upper()]["SEQ_LEN"], c=config[args.data.upper()]["NUM_CHANNELS"],
                            h=config[args.data.upper()]["HR_PATCH_SHAPE"][0] // config[args.data.upper()]["SCALE"],
                            w=config[args.data.upper()]["HR_PATCH_SHAPE"][1] // config[args.data.upper()]["SCALE"],
                            **model_settings[args.model_settings.upper()])

    #### Run task ####
    if args.task == "train":
        print("Training...")
        model = train(model=model, data_module=data_module, experiment_dir=experiment_dir, max_epochs=args.num_epochs, gpus=args.num_gpus, save=bool(args.model_save))
    
    elif args.task == "eval":
        print("Evaluating...")

        results_str = "{}\n".format(experiment_dir.upper())

        psnr, ssim = evaluate(model=model, eval_dataloader=data_module.train_dataloader())
        results_str += "TRAIN: PSNR: {} SSIM: {}\n".format(psnr, ssim)

        psnr, ssim = evaluate(model=model, eval_dataloader=data_module.val_dataloader())
        results_str += "VALID: PSNR: {} SSIM: {}\n".format(psnr, ssim)
        
        # TODO test data...

        print(results_str)

        # Save results to file
        with open(os.path.join(experiment_dir, "results.txt"), "w+") as f:
            f.write(results_str)

    elif args.task == "sample":
        print("Sampling...")
        sample_outputs(model=model, eval_dataloader=data_module.val_dataloader())

    print("Done")

#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