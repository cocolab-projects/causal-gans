"""
Stochastically generates images with at most two numbers
(4 and 3), adding noise to 4 from N(0,1) or U(0,1). (In
expectation,) a 4 with noise from N(0,1) (in expectation)
"causes" a 3 to appear, and a 4 with noise from U(0,1)
does not. (The choice of distributions is meant to be
arbitrary.)

@author mmosse19
@version June 2019
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transform
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
DATA_DIR = os.path.realpath(DATA_DIR)
# alternative:
# DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

SQR_DIM = 32
BLK_SQR = np.zeros((SQR_DIM,SQR_DIM))

C = 4					# cause
O = 4					# correlate
E = 3					# effect
p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

CAUSAL_NOISE_WT = 0.1
CORR_NOISE_WT = 0.1

"""
alt:

MNIST = torchvision.datasets.MNIST(
    download=True,
    root=".",
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
    ]),
    train=True
)
"""

def generate_worlds(mnist):
	act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
	cf_world = generate_cf_world(act_world)
	nums, utt = reformat(act_world)
	act_img = img(nums, utt, mnist)

def img(nums, utt, mnist):
	top_left, bottom_right = imgs_from_nums(nums, mnist)
	top_left = add_noise(top_left, utt)
	
	# put all four corner images together
	# consider switching axes for clarity
	img = np.concatenate((np.concatenate((top_left, BLK_SQR), axis=0),
                          np.concatenate((BLK_SQR, bottom_right), axis=0)),
                         axis=1)
	return plt.imshow(img)

def add_noise(top_left, utt):
	noise = BLK_SQR
	if "causes" in utt:
		noise = np.random.normal(255.0/2, 255.0/6, (SQR_DIM,SQR_DIM))*CAUSAL_NOISE_WT
	elif ("and" in utt):
		noise = np.random.uniform(0,255, (SQR_DIM,SQR_DIM))*CORR_NOISE_WT
	return np.minimum(np.add(noise, init_img), 255)

def imgs_from_nums(nums, mnist):
	top_left = nums[0] if num_from_mnist(nums[0], mnist) else BLK_SQR
	bottom_right = nums[1] if num_from_mnist(nums[2], mnist) else BLK_SQR
	return top_left, bottom_right

def num_from_mnist(digit):
	loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
	return mnist["digits"][loc]

def reformat(world):
	if(world["causal"]):
		cause = C if world["C"] else ""
		effect = E if (world["C"] and world["cE"]) else ""

		nums = [cause, effect]
		utt = str(cause) + ((" causes " + str(effect)) if str(effect) else "")
	else:
		corr = O if world["O"] else ""
		effect = E if world["bE"] else ""

		nums = [corr, effect]
		utt = str(corr) + ((" and " + str(effect)) if str(effect) else "")
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

    mnist_root = os.path.join(DATA_DIR, 'mnist')
    if not os.path.isdir(mnist_root):
        os.makedirs(mnist_root)

    train_mnist, test_mnist = load_mnist(mnist_root)
    mnist = test_mnist if args.test else train_mnist
    generate_worlds(mnist)
