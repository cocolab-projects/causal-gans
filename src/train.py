"""
train.py

@author mmosse19
@version July 2019
"""
import pdb

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from datasets import CausalMNIST
from models import LogisticRegression
from utils import (AverageMeter, save_checkpoint, free_params, frozen_params)

def loop(loader, model, mode, pbar=None):
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(loader):
        images,labels = images.to(device), labels.to(device).float()
        outputs = model(images)

        # if testing for accuracy, round outputs; else add dim to labels
        if (mode=="test"):
            outputs = np.rint(outputs.numpy().flatten())
        else:
            labels = labels.unsqueeze(1)

        # apply criterion
        criterion = torch.nn.BCELoss()
        loss = outputs == labels.numpy() if mode == "test" else criterion(outputs, labels)
        loss_amt = np.mean(loss) if mode == "test" else loss.item()

        loss_meter.update(loss_amt, loader.batch_size)

        if (mode == "train"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss_meter.avg})
            pbar.update()

    return loss_meter.avg

def run(loader, model, mode, epoch=0):
    if (mode == "train"):
        model.train()
        pbar = tqdm(total=len(loader))
        avg_loss = loop(loader, model, mode, pbar)
        pbar.close()
    else:
        model.eval()
        with torch.no_grad():
            avg_loss = loop(loader, model, mode)

    if (mode=="test"):
            print('====> test accuracy: {}%'.format(avg_loss*100))
    elif epoch % 20 == 0:
            print('====> {} epoch: {}\tloss: {:.4f}'.format(mode, epoch, avg_loss))
    return avg_loss

def load_checkpoint(folder='./', filename='model_best.pth.tar'):
    checkpoint = torch.load(folder + filename)
    epoch = checkpoint['epoch']
    track_loss = checkpoint['track_loss']
    model = checkpoint['model']
    return epoch, track_loss, model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='where to save checkpoints',default="./")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size [default=32]')
    parser.add_argument('--lr_rt', type=float, default=2e-4,
                        help='learning rate [default: 2e-4]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs [default: 200]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset = CausalMNIST()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    valid_dataset = CausalMNIST(split="validate")
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)

    log_reg = LogisticRegression()
    log_reg = log_reg.to(device)
    optimizer = torch.optim.Adam(log_reg.parameters(), lr=args.lr_rt)

    best_loss = float('inf')
    track_loss = np.zeros((args.epochs, 2))

    for epoch in range(int(args.epochs)):

        train_loss = run(train_loader, log_reg, "train", epoch)
        validate_loss = run(valid_loader, log_reg, "validate", epoch)

        is_best = validate_loss < best_loss
        best_loss = min(validate_loss, best_loss)
        track_loss[epoch, 0] = train_loss
        track_loss[epoch, 1] = validate_loss
        
        save_checkpoint({
            'epoch': epoch,
            'model': log_reg.state_dict(),
            'optimizer': optimizer.state_dict(),
            'track_loss': track_loss,
            'cmd_line_args': args,
            'seed': args.seed
        }, is_best, folder = args.out_dir)
        # np.save(os.path.join(args.out_dir, 'loss.npy'), track_loss)

    test_dataset = CausalMNIST(split='test')
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    test_model = LogisticRegression()
    _,_,state_dict = load_checkpoint(folder=args.out_dir)
    test_model.load_state_dict(state_dict)
    run(test_loader, test_model, "test")
