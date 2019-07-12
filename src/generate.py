"""
generate.py

@author mmosse19
@version June 2019
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import (clamp_img, standardize_img, viewable_img, MIN_COLOR, MAX_COLOR, MNIST_MIN_COLOR, MNIST_MAX_COLOR)

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
from torchvision.utils import save_image

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")

NUM1 = 4
NUM2 = 3

p = {"causal": .5, "C": .8, "noC": .8, "cE": .6, "bE": .5}
event_functions = { "causal": lambda img: warp(img, 1, 1),
                    "C": lambda img: warp(img, 1, 1),
                    "noC": lambda img: img,
                    "cE": lambda img: warp(img, 1, 1),
                    "bE": lambda img: img
                    }
TOTAL_NUM_WORLDS = len(p) + 1

CORNER_DIM = 32
IMG_DIM = CORNER_DIM*2
BLK_SQR = np.zeros((CORNER_DIM,CORNER_DIM))

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
    return mnist["digits"][loc]

def imgs_of_worlds(worlds, mnist):
    four_img, three_img = num_from_mnist(NUM1, mnist), num_from_mnist(NUM2, mnist)
    return [img_of_world(world, four_img, three_img) for world in worlds]

# note: BLK_SQR is all 0's; assumes no preprocessing
def img_of_world(world, four_img, three_img):
    nums, utt = reformat(world)[0], reformat(world)[1]

    # set numbers in corners if they're in the world
    top_left = four_img if nums[0] else BLK_SQR
    bottom_right = three_img if nums[1] else BLK_SQR
    
    # apply nonlinear transformations; top_left is in interval [0,1]
    for event in world:
        if (event): top_left = event_functions[event](top_left)
    top_left = clamp_img(top_left, MNIST_MIN_COLOR, MNIST_MAX_COLOR)

    # put all four corner images together
    img = np.concatenate((np.concatenate((top_left, BLK_SQR), axis=1),
                      np.concatenate((BLK_SQR, bottom_right), axis=1)),
                     axis=0)
    return standardize_img(img, from_unit_interval=True)

def reformat(world):
    if(world["causal"]):
        num1 = NUM1 if world["C"] else ""
        num2 = NUM2 if ((world["C"] and world["cE"]) or world["bE"]) else ""

        nums = [num1, num2]
        utt = (("causal " + str(num1)) if str(num1) else "") + str(num2)
    else:
        num1 = NUM1 if world["noC"] else ""
        num2 = NUM2 if world["bE"] else ""

        nums = [num1, num2]
        utt = str(num1) + str(num2)
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

    # TODO: only get relevant digit values.
    train_data = {
        'digits': np.concatenate([imgs.numpy()*MAX_COLOR for imgs,labels in train_loader], axis = 0)[:,0,...],
        'labels': np.concatenate([labels.numpy() for imgs,labels in train_loader], axis = 0),
    }
    test_data = {
        'digits': np.concatenate([imgs.numpy()*MAX_COLOR for imgs, labels in test_loader],axis = 0)[:,0,...],
        'labels': np.concatenate([labels.numpy() for imgs,labels in train_loader], axis = 0),
    }

    return train_data, test_data

# might be worth using pathlib
def mnist_dir_setup(train):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_mnist, test_mnist = load_mnist(DATA_DIR)
    return train_mnist if train else test_mnist

def warp(img, A, B):
    new_img = copy.deepcopy(img)
    for u in range(len(img)):
        for v in range(len(img[0])):
            x = int(u + A *np.sin(2.0 * np.pi  * v / B))
            new_img[x][v] = img[u][v]
    return new_img

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

    img = mnist["digits"][0]