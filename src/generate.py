"""
generate.py

Based on code by mhw32

@author mmosse19
@version July 2019
"""
import os
import copy
import itertools
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
# TODO: when to apply functions?
event_functions = { "causal": lambda img: warp(img, 1, 1),
                    "C": lambda img: warp(img, 1, 1),
                    "noC": False,
                    "cE": lambda img: warp(img, 1, 1),
                    "bE": False
                    }
VARIABLES = list(p.keys())
TOTAL_NUM_WORLDS = len(p) + 1

CORNER_DIM = 32
IMG_DIM = CORNER_DIM*2
BLK_SQR = np.zeros((CORNER_DIM,CORNER_DIM))

def warp(img, A, B):
    new_img = copy.deepcopy(img)
    for old_x in range(CORNER_DIM):         # int(len(img)/2)
        for y in range(CORNER_DIM):         # int(len(img[0])/2)
            x = int(old_x + A *np.sin(2.0 * np.pi  * y / B))
            new_img[x][y] = img[old_x][y]
    return new_img

def enforce_rules(world):
    if (world["C"] and not world["causal"]):
        world["causal"] = False
    if (world["cE"] and not world["C"]):
        world["cE"] = False

def generate_worlds(mnist, n, cf = False, transform=True):
    num1_locs = np.where(mnist["labels"] == NUM1)[0]
    num2_locs = np.where(mnist["labels"] == NUM2)[0]

    scenarios = [] # a scenario is an actual world and its cfs
    for i in range(n):
        act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
        enforce_rules(act_world)

        if (cf):
            cf_worlds = generate_cf_worlds(act_world)
            all_worlds = [act_world] + cf_worlds    # act_world comes first
            imgs = imgs_of_worlds(all_worlds, mnist, transform, num1_locs, num2_locs)
            img = np.concatenate(tuple(imgs), axis=1)
            label = reformat(act_world)[1]
        else:
            all_worlds = [act_world]
            img = imgs_of_worlds([act_world], mnist, transform, num1_locs, num2_locs)[0]
            label = reformat(act_world)[1]
        
        scenarios.append((img, label))

        # save_image(torch.from_numpy(joined_img), str(i) + ".jpg")
    return scenarios

def num_from_mnist(mnist, locs):
    loc = np.random.choice(locs)
    return mnist["digits"][loc]

def imgs_of_worlds(worlds, mnist, transform, num1_locs, num2_locs):
    num1_img = num_from_mnist(mnist, num1_locs)
    num2_img = num_from_mnist(mnist, num2_locs)    
    return [img_of_world(world, num1_img, num2_img, transform) for world in worlds]

# note: BLK_SQR is all 0's; assumes no preprocessing (else, BLK_SQR would be all -1's)
def img_of_world(world, num1_img, num2_img, transform):
    nums, utt = reformat(world)[0], reformat(world)[1]

    # set numbers in corners if they're in the world
    top_left = num1_img if nums[0] else BLK_SQR
    bottom_right = num2_img if nums[1] else BLK_SQR
   
    # apply nonlinear transformations; note that pixels are in interval [0,1]
    if (transform):
        for event in world:
            if (event_functions[event] and world[event]): top_left = event_functions[event](top_left)
        top_left = clamp_img(top_left, MNIST_MIN_COLOR, MNIST_MAX_COLOR)

    # put all four corner images together
    img = np.concatenate((np.concatenate((top_left, BLK_SQR), axis=1),
                      np.concatenate((BLK_SQR, bottom_right), axis=1)),
                     axis=0)

    # move image from [0,1] to [-1,1]
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
    return [enforce_rules(flip_rvs(act_world, key_to_vary)) for key_to_vary in act_world]

def all_possible_imgs(start_img):
    all_functions = [fn for fn in event_functions.values() if fn]
    all_function_combinations = []

    for values in list(itertools.product([0,1], repeat=len(all_functions))):
        combo = dict([(fn, value) for fn,value in zip(all_functions, values)])
        all_function_combinations.append(combo)

    # combo is a tuple, saying which functions to apply (e.g., (1,0,0,1,0))
    for i, combo in enumerate(all_function_combinations):
        img = copy.deepcopy(start_img)

        for fn, fn_indicator in combo.items():
            if (fn_indicator):
                img = fn(img)

        img = clamp_img(img, MNIST_MIN_COLOR, MNIST_MAX_COLOR)
        save_image(torch.from_numpy(img), "modified " + str(i) + ".png")

    breakpoint()

# images' pixels are in interval [0,1]
def load_mnist(root, test):
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
            train=False,
            download=True,
            transform=transform.Compose([transform.Resize(CORNER_DIM), transform.ToTensor()])
        ),
        batch_size = 500,
        shuffle=False
    )
    

    (loader, data_kind) = (test_loader, "test") if test else (train_loader, "train/validate")

    print("retrieving " + data_kind + " data from mnist...")
    data = {
        'digits': np.concatenate([imgs.numpy() for imgs,labels in loader], axis = 0)[:,0,...],
        'labels': np.concatenate([labels.numpy() for imgs,labels in loader], axis = 0),
    }
    print("retrieved " + data_kind + " data from mnist.")
    return data

# might be worth using pathlib
def mnist_dir_setup(test):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    return load_mnist(DATA_DIR, test)

if __name__ ==  "__main__":
    import argparse
    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [default: 42]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='sample digits from test set of MNIST [default: False]')
    parser.add_argument('--dataset_size', type=int, default=10,
                        help='number of images in dataset [default: 10]')
    args = parser.parse_args()

    np.random.seed(args.seed)

    mnist= mnist_dir_setup(args.test)