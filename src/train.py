import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from datasets import CausalMNIST
from models import LogisticRegression

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
    valid_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    log_reg = LogisticRegression()
    log_reg = log_reg.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(log_reg.parameters(), lr=args.lr_rt)

    for epoch in range(int(args.epochs)):
        pbar = tqdm(total=len(train_loader))
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = log_reg(images)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_postfix({'loss': loss.item()})
        pbar.close()

    # plt.plot(loss_data)
    # plt.show()

