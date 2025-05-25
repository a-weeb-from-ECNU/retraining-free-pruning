import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    set_seed,
)
from hardPrune.vit.vit_prune_model import getPrunedViTModel
from evaluate.vision import test_accuracy_vit2


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=[
    "cifar10",
    "cifar100", 
    "imagenet",
    "food101",
    "oxford_flowers102",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--metric", type=str, choices=[
    "mac",
    "latency",
], default="mac")
parser.add_argument("--constraint", type=float, required=True,
    help="MAC/latency constraint relative to the original model",
)
parser.add_argument("--mha_lut", type=str, default=None)
parser.add_argument("--ffn_lut", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--drop_rearrange", action="store_true",
    help="Whether to skip the rearrangement step", default=False
)
parser.add_argument("--drop_rescale", action="store_true",
    help="Whether to skip the rescaling step", default=False
)


def main():
    args = parser.parse_args()
    IS_LARGE = "large" in args.model_name
    img_size = 224
    # For ViT, sequence length is determined by patch size and image size
    # Default ViT-Base: 16x16 patches on 224x224 image = 196 patches + 1 [CLS] = 197
    seq_len = (img_size // 16) ** 2 + 1  # Assuming patch size of 16

    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            args.metric,
            str(args.constraint),
            f"seed_{args.seed}"
        )
    if args.drop_rearrange:
        args.output_dir += "/no_rearrange"
    elif args.drop_rescale:
        args.output_dir += "/no_rescale"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned ViT model and the corresponding image processor
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    model = getPrunedViTModel()
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        use_auth_token=None,
        use_fast=True
    )
    
    # Prepare the model
    model = model.cuda()
    model.eval()
 
    for param in model.parameters():
        param.requires_grad_(False)
    

    
    test_acc = test_accuracy_vit2(model,None, None, image_processor, args.task_name)
    logger.info(f"{args.task_name} Test accuracy: {test_acc:.4f}")
    
    
if __name__ == "__main__":
    main()