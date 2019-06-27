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
import torch
import torch.utils.data

SQR_DIM = 28
BLK_SQR = np.zeros((SQR_DIM,SQR_DIM))

C = "4 with gaussian"	# cause
O = "4 with uniform"	# correlate
E = 3					# effect
p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

def generate_vec():
	world = {key: np.random.binomial(1,p[key]) for key in p.keys()}
	nums, utt = reformat(world)
	# return image(nums, utt) # todo: turn C/O into ints for processing

def image(nums, utt):
	# fill in top left and bottom right corners
	cause_img = img_from_pos(nums[0])
	caused_img = img_from_pos(nums[-1])
	
	# add noise
	noise = BLK_SQR
	if "causes" in utt:
		noise = np.random.normal((SQR_DIM,SQR_DIM))
	elif ("and" in utt):
		noise = np.random.uniform(0,255, (SQR_DIM,SQR_DIM))
	cause_img = np.minimum(np.add(noise, cause_img), 255)

	# put all four corner images together
	img = np.concatenate((np.concatenate((cause_img, BLK_SQR), axis=0),
                          np.concatenate((BLK_SQR, caused_img), axis=0)),
                         axis=1)
	return img

def img_from_pos(digit):
	return img_from_mnist(digit) if digit else BLK_SQR

def img_from_mnist(digit):
	loc = np.random.choice(np.where(mnist["labels"] == digit)[0])
	return mnist["digits"][loc]
"""
[pos1, pos2]
[pos3, pos4]

"""
def reformat(world):
	if(world["causal"]):
		pos1 = str(C) if world["C"] else ""
		pos4 = str(E) if world["C"] and world["cE"] else ""
		
		nums = [pos1, "", "", pos4]
		utt = pos1 + (("causes " + pos4) if pos4)
		return nums, utt
	else:
		pos1 = str(O) if world["O"] else ""
		pos4 = str(E) if E if world["bE"] else ""
		
		nums = [pos1, "", "", pos4]
		utt = pos1 + (("and " + pos4) if pos4)
