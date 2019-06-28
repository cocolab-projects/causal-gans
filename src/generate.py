"""
generate.py

@author mmosse19
@version June 2019
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
import torchvision.utils as utils

import pdb

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
DATA_DIR = os.path.realpath(DATA_DIR)
# alternative:
# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

C = 4					# cause
O = 4					# correlate
E = 3					# effect
p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

SQR_DIM = 28
BLK_SQR = np.zeros((SQR_DIM,SQR_DIM))
MAX_COLOR = 255.0

CAUS_NOISE_WT = 0.01
CORR_NOISE_WT = 0.005
CAUS_NOISE_MEAN = MAX_COLOR/2
CAUS_NOISE_VARIANCE = (MAX_COLOR/6)**2

def generate_worlds(mnist):
	for i in range(10):
		act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
		cf_world = generate_cf_world(act_world)
		nums, utt = reformat(act_world)
		
		act_img = img(nums, utt, mnist)
		name = "img " + str(i) + "_" + utt + ".jpg"
		utils.save_image(torch.from_numpy(act_img), name)

def img(nums, utt, mnist):
	top_left, bottom_right = imgs_from_nums(nums, mnist)
	top_left = add_noise(top_left, utt)
	
	# put all four corner images together
	# consider switching axes for clarity
	return np.concatenate((np.concatenate((top_left, BLK_SQR), axis=0),
                          np.concatenate((BLK_SQR, bottom_right), axis=0)),
                         axis=1)

def add_noise(init_img, utt):
	noise = BLK_SQR
	if "causal" in utt:
		noise = np.random.normal(CAUS_NOISE_MEAN, CAUS_NOISE_VARIANCE, (SQR_DIM,SQR_DIM))*CAUS_NOISE_WT
	elif ("4" in utt):
		noise = np.random.uniform(0,MAX_COLOR, (SQR_DIM,SQR_DIM))*CORR_NOISE_WT
	return np.minimum(np.add(noise, init_img), MAX_COLOR)

def imgs_from_nums(nums, mnist):
	top_left = num_from_mnist(nums[0], mnist) if nums[0] else BLK_SQR
	bottom_right = num_from_mnist(nums[1], mnist) if nums[1] else BLK_SQR
	return top_left, bottom_right

def num_from_mnist(digit, mnist):
	loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
	return mnist["digits"][loc]

def reformat(world):
	if(world["causal"]):
		cause = C if world["C"] else ""
		effect = E if (world["C"] and world["cE"]) else ""

		nums = [cause, effect]
		utt = (("causal " + str(cause)) if str(cause) else "") + (str(effect) if str(effect) else "")
	else:
		corr = O if world["O"] else ""
		effect = E if world["bE"] else ""

		nums = [corr, effect]
		utt = str(corr) + (str(effect) if str(effect) else "")
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
def generate_cf_world(act_world):
	cfs = [flip_rvs(act_world, key_to_vary) for key_to_vary in act_world]
	# why access [0]?
	cf_worlds = [reformat(w)[0] for w in cfs]
	return cf_worlds

"""

"""
def load_mnist(root):
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([
                transform.Resize(32),
            ])
        )
    )

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform.Compose([
                transform.Resize(32),
            ])
        )
    )
   
    train_data = {
        'digits': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy(),
    }
    test_data = {
        'digits': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy(),
    }

    return train_data, test_data

"""

"""
if __name__ ==  "__main__":
	# handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
	                    help='random seed [default: 42]')
    parser.add_argument('--test', action='store_true', default=False,
                        help='sample digits from test set of MNIST [default: False]')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # setup mnist
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    mnist_root = DATA_DIR

    train_mnist, test_mnist = load_mnist(mnist_root)
    mnist = test_mnist if args.test else train_mnist
    generate_worlds(mnist)
