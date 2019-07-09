"""
generate.py

@author mmosse19
@version June 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (AverageMeter, save_checkpoint)

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
from torchvision.utils import save_image

import pdb

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")

# interpretations
# "causal": probability that weather forecast predicts rain (it never predicts no rain when there's rain)
# "C": probability of rain, given forecast
# "noC": probability of sprinklers
# note: O and C are indistinguishable; both are evidenced by wet grass
# "CE": probability of roof being wet, due to rain
# "bE": probability of mike throwing water balloon at roof

p = {"causal": .5, "C": .8, "noC": .8, "cE": .6, "bE": .5}
event_functions = { "causal": lambda x: x**2,
                    "C": lambda x: (x+1)**2,
                    "noC": lambda x: x,
                    "cE": lambda x: (x+2)**2,
                    "bE": lambda x: x
                    }
TOTAL_NUM_WORLDS = len(p) + 1

CORNER_DIM = 32
IMG_DIM = CORNER_DIM*2
BLK_SQR = np.zeros((CORNER_DIM,CORNER_DIM))

MIN_COLOR = 0.0
MAX_COLOR = 255.0

# why do we vary "causal"?
# causal entity or process
# to do: clean up handling of case where cf
def generate_worlds(mnist, n=1, cf = False, transform=True):

    scenarios = [] # a scenario is an actual world and its cfs
    for i in range(n):
        act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
        if (cf):
            cf_worlds = generate_cf_worlds(act_world)
            all_worlds = [act_world] + cf_worlds    # act_world comes first
            imgs = imgs_of_worlds(all_worlds, mnist)
            img = np.concatenate(tuple(imgs), axis=1)
            labels = [reformat(world)[1] for world in all_worlds]
            label = labels[0]
        else:
            all_worlds = [act_world]
            img, label = imgs_of_worlds([act_world], mnist)[0], reformat(act_world)[1]
        
        scenarios.append((img, label))
        if (n is 1): return scenarios[0]

        # save_image(torch.from_numpy(joined_img), str(i) + ".jpg")
    return scenarios

def num_from_mnist(digit, mnist):
    loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
    # loc = (np.where(mnist["labels"] == digit)[0])[0]
    return mnist["digits"][loc]

def imgs_of_worlds(worlds, mnist):
    four_img, three_img = num_from_mnist(4, mnist), num_from_mnist(3, mnist)
    return [img_of_world(world, four_img, three_img) for world in worlds]

def img_of_world(world, four_img, three_img):
    nums, utt = reformat(world)[0], reformat(world)[1]

    # set numbers in corners if they're in the world
    top_left = four_img if nums[0] else BLK_SQR
    bottom_right = three_img if nums[1] else BLK_SQR
    
    # apply nonlinear transformations
    for event in world:
        if (event): top_left = event_functions[event](top_left)
    top_left = np.maximum(np.minimum(top_left, MAX_COLOR), MIN_COLOR)

    # put all four corner images together
    return np.concatenate((np.concatenate((top_left, BLK_SQR), axis=1),
                      np.concatenate((BLK_SQR, bottom_right), axis=1)),
                     axis=0)

def reformat(world):
    if(world["causal"]):
        four = 4 if world["C"] else ""
        three = 3 if ((world["C"] and world["cE"]) or world["bE"]) else ""

        nums = [four, three]
        utt = (("causal " + str(four)) if str(four) else "") + str(three)
    else:
        four = 4 if world["noC"] else ""
        three = 3 if world["bE"] else ""

        nums = [four, three]
        utt = str(four) + str(three)
    return nums, utt

def flip_rv(actual, key, key_to_vary):
    a = actual[key]
    return not a if key == key_to_vary else a

"""
Given an actual set of values for the random variables, flip exactly one of
those.
"""
def flip_rvs(act_world, key_to_vary):
    return {key: flip_rv(act_world, key, key_to_vary) for key in act_world}

"""
Given an actual set of values for the random variables, generate a list of
counterfactuals by flipping each of the random variables, one at a time
"""
def generate_cf_worlds(act_world):
    return [flip_rvs(act_world, key_to_vary) for key_to_vary in act_world]

# this takes a lot of time and should be improved
def load_mnist(root):
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([transform.Resize(CORNER_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([transform.Resize(CORNER_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )

    train_data = {
        'digits': np.concatenate([img.numpy()*MAX_COLOR for img,label in train_loader], axis = 0)[:,0,...],
        'labels': np.concatenate([label.numpy() for img,label in train_loader], axis = 0),
    }
    test_data = {
        'digits': np.concatenate([img.numpy()*MAX_COLOR for img, label in test_loader],axis = 0)[:,0,...],
        'labels': np.concatenate([label.numpy() for img,label in train_loader], axis = 0),
    }

    return train_data, test_data

# might be worth using pathlib
def mnist_dir_setup(train):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_mnist, test_mnist = load_mnist(DATA_DIR)
    return train_mnist if train else test_mnist

if __name__ ==  "__main__":
    import argparse
    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--train', action='store_true', default=True,
                        help='sample digits from train set of MNIST [default: True]')
    parser.add_argument('--dataset_size', type=int, default=10,
                        help='number of images in dataset [default: 10]')
    args = parser.parse_args()

    np.random.seed(args.seed)

    mnist= mnist_dir_setup(args.train)

    generate_worlds(mnist, n=args.dataset_size, cf=True)
