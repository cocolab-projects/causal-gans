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
import numpy as np

import pytorch

SQR_DIM = 28
BLK_SQR = np.zeros((SQR_DIM,SQR_DIM))

C = 4					# cause
O = 4					# correlate
E = 3					# effect
p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

def generate_worlds():
	act_world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
	cf_world = generate_cf_world(act_world)
	nums, utt = reformat(act_world)
	act_img = img(nums, utt)

def img(nums, utt):
	# fill in top left and bottom right corners
	init_img = img_from_pos(nums[0])	# cause or correlate
	effect_img = img_from_pos(nums[-1])
	
	# add noise
	noise = BLK_SQR
	if "causes" in utt:
		noise = np.random.normal((SQR_DIM,SQR_DIM))
	elif ("and" in utt):
		noise = np.random.uniform(0,255, (SQR_DIM,SQR_DIM))
	init_img = np.minimum(np.add(noise, init_img), 255)

	# put all four corner images together
	img = np.concatenate((np.concatenate((init_img, BLK_SQR), axis=0),
                          np.concatenate((BLK_SQR, effect_img), axis=0)),
                         axis=1)
	return img

def img_from_pos(digit):
	return img_from_mnist(digit) if digit else BLK_SQR

def img_from_mnist(digit):
	loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
	return mnist["digits"][loc]
"""
[pos1, blnk]
[blnk, pos2]

"""
def reformat(world):
	if(world["causal"]):
		cause = str(C) if world["C"] else ""
		effect = str(E) if (world["C"] and world["cE"]) else ""

		nums = [cause, effect]
		utt = cause + ((" causes " + effect) if effect else "")
	else:
		corr = str(O) if world["O"] else ""
		effect = str(E) if world["bE"] else ""

		nums = [corr, effect]
		utt = corr + ((" and " + effect) if effect else "")
	return nums, utt

def flip_rv(actual, key, key_to_vary):
	a = actual[key]
	if (key == key_to_vary):
		return (not a)
	else:
		return a

"""
Given an actual set of values for the random variables, flip exactly one of
those .
"""
def flip_rvs(act_world, key_to_vary):
	return {key: flip_rv(act_world, key, key_to_vary) for key in act_world}

"""
Given an actual set of values for the random variables, generate a list of
counterfactuals by flipping each of the random variables, one at a time
"""
def generate_cf_world(act_world):
	cfs = [flip_rvs(act_world, key_to_vary) for key_to_vary in act_world]
	cf_vecs = [reformat(w)[0] for w in cfs]
	return cf_vecs

generate_worlds()
