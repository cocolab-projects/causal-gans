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

square_size = 4

C = 4
O = 4
E = 3

# params for image generation
# C is cause; E is effect
p = {"causal": .5, "C": .8, "O": .8, "cE": .9, "bE": .2}

def generate_vec():
	result = {key:np.random.binomial(1,p[key]) for key in p.keys()}
	vec = vec_from_result(result)
	return img

def vec_from_result(result):
	if(result["causal"]):
		return [C if result["C"] else "", "", "", E if result["cE"] else ""]
	else:
		return [O if result["O"] else "", "", "", E if result["bE"] else ""]
