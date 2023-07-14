#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils import write_params
from models.unet import ForwardRemover, ResRemover
from data.dataloader import ImageDataset

random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="CBSD68", help='name of the dataset')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--n_cpu', type=float, default=1, help='number of CPU to load data')
parser.add_argument('--description', type=str, default='train and test on CBSD68', help='CBSD68 train')
parser.add_argument('--test_interval', type=int, default=5, help='test interval')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='checkpoint_interval')
parser.add_argument('--save_num', type=int, default=0, help='label for saving')
parser.add_argument('--epoch', type=int, default=0, help='training start epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='training end epoch')
parser.add_argument('--loss', type=str, default='L2', help='L1 or L2 to be utilized')

opt = parser.parse_args()

path = './result/'+str(opt.save_num)+'/'

log_path = path+'log/'+opt.dataset_name+'/'
os.makedirs(path+'saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs(path+'output/%s' % opt.dataset_name, exist_ok=True)

logger = SummaryWriter(log_dir=log_path)
write_params(log_path, parser, opt.description)
    
model = ResRemover().to(device)
L1_criterion = torch.nn.L1Loss().to(device)
L2_criterion = torch.nn.MSELoss().to(device)

if opt.loss == 'L2':
    criterion = L2_criterion
elif opt.loss == 'L1':
    criterion = L1_criterion


optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

dataset = ImageDataset(dataset_path=r'./dataset/CBSD68',eval_mode=False)
train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)

dataset = ImageDataset(dataset_path=r'./dataset/CBSD68',eval_mode=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)
eval_samples = iter(test_loader)

def eval_metric(step):
    model.eval()
    batch = next(eval_samples)
    batch['label'] = batch['label'].to(device)
    batch['image'] = batch['image'].to(device)
    batch['image'].requires_grad = True

    output = model(batch['image'])
    l1_loss = L1_criterion(output, batch['label'])
    l2_loss = L2_criterion(output, batch['label'])
    logger.add_scalar('test/L1_loss', l1_loss.item(), step)
    logger.add_scalar('test/L2_loss', l2_loss.item(), step)


    sample = torch.cat((batch['label'].data, batch['image'].data), -1)
    sample = torch.cat((sample.data, output.data), -1)

    # save_image(sample, path+'images/%s_img.png' % (step), nrow=4, normalize=True)
    logger.add_image('img/sample%d'%(step), sample[0,:,:,:], 0, dataformats='CHW')
    # logger.add_image('img/sample', sample, step)
    model.train()
    
total_step = 0
print('Start to train ...')
print('Device:',device)

for epoch in range(opt.epoch, opt.n_epochs):
    print(f"epoch: {epoch}")
    bar = enumerate(train_loader)
    length = len(train_loader)
    bar = tqdm(bar, total=length)
    for i, batch in bar:
        total_step += 1

        batch['label'] = batch['label'].to(device)
        batch['image'] = batch['image'].to(device)
        batch['image'].requires_grad = True

        output = model(batch['image'])
        
        optimizer.zero_grad()
        loss = criterion(output, batch['label'])
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()

        logger.add_scalar('train/loss', loss.item(), total_step)
        
        if total_step % opt.test_interval == 0:
            try:
                eval_metric(total_step)
            except StopIteration:
                eval_samples = iter(test_loader)
                eval_metric(total_step)
        
        if total_step % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), path+'saved_models/model_%d.pth'%(total_step))