import standardPerceptron as pcp
import imp
imp.reload(pcp)
import pandas as pd 
import numpy as np

def XY4setosa(data):   # data is a DataFrame
	"""get x, y for classifying setosa """
	label = np.array(data.iloc[:,4])
	labelConvert = []
	for _ in label:
		if _ == "Iris-setosa":   # 0 means setosa
		    _ = 0
		    labelConvert.append(_)
		else:  
		    _ = 1
		    labelConvert.append(_)
	data.drop(data.columns[len(data.columns)-1],axis=1)   # drop the last column !!!!!
	data.loc[:, data.columns[4]] = labelConvert

	X = np.array(data.iloc[:,0:4])   # input
	Y = np.array(data.iloc[:,4])    # desired output
	return X, Y


def XY4virginica(data): # data is a DataFrame
	"""get x, y for classifying virginica """
	label = np.array(data.iloc[:,4])
	labelConvert = []
	for _ in label:
		if _ == "Iris-virginica":   # 0 means virginica
		    _ = 0
		    labelConvert.append(_)
		else:  
		    _ = 1
		    labelConvert.append(_)
	data.drop(data.columns[len(data.columns)-1],axis=1)   # drop the last column !!!!!
	data.loc[:, data.columns[4]] = labelConvert

	# X = np.array(data.iloc[:,0:4])   # input
	Y = np.array(data.iloc[:,4])    # desired output
	return Y  # X is identical to setosa


def getO1(x,y):
	"""instanciate a perceptron, learn from dataset for classifying
	setosa and non-setosa,
	Return The output of this perceptron,
	The input of Gates as well!!!!!"""
	per = pcp.Perceptron(x,y)
	per.learn4setosa(eta=0.01, showProcess=False)
	print("What this perceptron has learned:")
	print("bias: {:.3f}".format(per.bias))
	print("weights:", ", ".join("{:.3f}".format(_) for _ in per.weights))
	o1 = per.outAsIn(x)
	return np.array(o1), per.weights, per.bias

def getO2(x,y):
	"""instanciate a perceptron, learn from virginica for classifying
	virginica and non-virginica.
	Return The output of this perceptron,
	The input of Gates as well!!!!!"""
	per = pcp.Perceptron(x,y)
	per.learn4virginica(eta=0.005, showProcess=False)
	print("What this perceptron has learned:")
	print("bias: {:.3f}".format(per.bias))
	print("weights:", ", ".join("{:.3f}".format(_) for _ in per.weights))
	# print(per.bias, per.weights)
	o2 = per.outAsIn(x)
	return np.array(o2), per.weights, per.bias 


def buildO(o1,o2):
	"""build two inputs signals in one array"""
	o1 = o1.reshape((o1.shape[0],1))
	o2 = o2.reshape((o2.shape[0],1))
	return np.hstack((o1,o2))



