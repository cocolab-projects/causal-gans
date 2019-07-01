import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    train_dataset = CausalMNIST()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, )

    log_reg = LogisticRegression()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(log_reg.parameters(), lr=args.lr_rt)

    loss_data = []
    for epoch in range(int(args.epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = Variables(images.view())
            labels = Variable(labels)

            otpimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_data.append(loss)
            loss.backward()
            optimizer.step()

    plot(loss_data)
    plt.show()

