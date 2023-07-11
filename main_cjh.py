import os, sys
import argparse
import random
import logging
import datetime
import pathlib
import wandb
import torch
import numpy as np


def train(args):
    pass

def evaluate(args):
    pass

def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Random Seed
    parser.add_argument('--seed', type=int, default=88245, help='Random seed.')

    
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training the model")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial Learning Rate')
    
    # # Optimization Parameters
    # parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    # parser.add_argument('--lr_decay', type=int, default=1, help="Wheter to perform LR decay on plateau.")
    # parser.add_argument('--decay_factor', type=float, default=0.5, help="LR decay factor.")
    # parser.add_argument('--decay_patience', type=int, default=5, help="LR decay patience.")
    # parser.add_argument('--num_decays', type=int, default=3, help="Number of LR decays before the training halt.")
    # parser.add_argument('--l2_reg', type=float, default=1e-4, help="L2 Deacy factor for weight regularization.")

    # Model type
    parser.add_argument('--model_type', type=str, default='AttGlobal_Scene_CAM_NFDecoder', help="A | B | C")
    
    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=20, help="")
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU IDs for model running")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id