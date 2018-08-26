import numpy as np 

def Gate1(x):
	"""
	0 1 | 0
	1 0 | 0 
	1 1 | 1
	"""
	weights = [1,1]
	bias = -1.5
	return sgn(np.add(np.dot(x, weights), bias))

def Gate2(x):
	"""
	0 1 | 1
	1 0 | 0
	1 1 | 0
	"""
	weights = [-1,1]
	bias = -0.5
	return sgn(np.dot(x, weights)+ bias)

def Gate3(x):
	"""
	0 1 | 0
	1 0 | 1
	1 1 | 0
	"""
	weights=[1,-1]
	bias = -0.5
	return sgn(np.dot(x, weights) + bias)

    
def sgn(y):      
	if (y>0):
	    return 1
	elif (y<0):
	    return 0

def groupOutput(x):  # x is a nX2 ndarray
	"""group the output and pass it to gates"""
	gateOutput = []
	for _ in x:
		gateOutput.append([Gate1(_), Gate2(_), Gate3(_)])
	return gateOutput

def forTestSample(sample, w1,b1,w2,b2):
	tmp1 = sgn(np.dot(w1,sample) + b1)
	tmp2 = sgn(np.dot(w2,sample) + b2)
	return [Gate1([tmp1,tmp2]), Gate2([tmp1,tmp2]),Gate3([tmp1,tmp2])]