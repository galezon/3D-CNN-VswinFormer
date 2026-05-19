import argparse
import logging
import torch
from pathlib import Path
from model import swin_tiny_patch4_window7_224 as create_model
from train import train
from logging_config import setup_logging
from config import BATCH_SIZE, NO_WORKERS
from register import train_ready_registered_datasets

from config import (
    PRETRAIN_WEIGHTS_PATH,
    NO_WORKERS
)

from dataset import get_loaders

logger = logging.getLogger(__name__)


def main(args):

    output_dir = args.output_dir
    epochs = args.epochs
    dataset = args.dataset
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    no_workers = args.no_workers if args.no_workers is not None else NO_WORKERS
    resume_from = args.checkpoint_path

    setup_logging(output_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} device")
    
    logger.info(f"Using {NO_WORKERS} dataloader workers")
    train_loader, val_loader, train_num, val_num = get_loaders(batch_size, no_workers, dataset)

    logger.info(f"Using {train_num} images for training, {val_num} images for validation")    

    # TODO: move loading of the model to model.py
    # load weights
    weights = PRETRAIN_WEIGHTS_PATH
    predict_model = torch.load(weights)
    predict_model['state_dict'] = {key.replace('backbone.', ''): value for key, value in predict_model['state_dict'].items()}
    
    for k in list(predict_model['state_dict'].keys()):  # Match the weight of the video swin Transformer
        if "patch_embed" in k:
            del predict_model['state_dict'][k]

    model = create_model(num_classes=2, init_weights=True).to(device)
    net_dict = model.state_dict()
    #print(net_dict.keys())
    #print(predict_model['state_dict'].items())

    state_dict = {k: v for k, v in predict_model['state_dict'].items() if k in net_dict.keys()}

    logger.info(f"Loaded pretrained weights: {len(state_dict)} layers matched")
    net_dict.update(state_dict)  
    model.load_state_dict(net_dict) 
    logger.info("Model loaded successfully")

    train(
        model,
        device,
        train_loader,
        val_loader,
        train_num,
        val_num,
        output_dir,
        epochs,
        resume_from
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)  # example
    parser.add_argument("--epochs", type=int, default=550)
    parser.add_argument("--dataset", type=str, required=True, choices=train_ready_registered_datasets)
    parser.add_argument("--batch_size", type=int, required=False)  # default in config
    parser.add_argument("--no_workers", type=int, required=False)  # default in config
    parser.add_argument("--checkpoint_path", type=str, default=None)


    args = parser.parse_args()

    main(args)