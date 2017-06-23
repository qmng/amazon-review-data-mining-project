import pandas
import sys
import numpy as np
import tensorflow_wrapper as tw
import process
import performance
from collections import Counter

def printNonZero(d):
	# Prints the key:value pairs of the dictionary that are non-zero
	for key in d:
		if d[key] != 0:
			print(key, d[key])

def printThreshold(d, t):
	# Prints the key:value pairs of the dictionary with value higher than a threshold t
	for key in d:
		if d[key] >= t:
			print(key, d[key])

def clusterPrepare(product, numEpochs, alpha, delta):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_"+product+".csv"
	#Create paths of training and testing reviews
	trainpath = "train_"+product+".csv"
	testpath = "test_"+product+".csv"

	#Read reviews
	df_train = pandas.read_csv(trainpath)
	df_test = pandas.read_csv(testpath)
	df = pandas.read_csv(allpath)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words using word2vec.txt
	trainX = np.array(process.formatClusterX(df_train, wordList, delta))
	testX = np.array(process.formatClusterX(df_test, wordList, delta))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))
	
	return [trainX, trainY, testX, testY, numEpochs, alpha]

def clusterDemo(product, numEpochs, alpha, delta):
	params = clusterPrepare(product, numEpochs, alpha, delta)
	tw.regression(params[0], params[1], params[2], params[3], params[4], params[5])

def clusterVector(product, numEpochs, alpha, delta):
	params = deltaPrepare(product, numEpochs, alpha, delta)
	[s, weights, bias] = tw.getTrainingSession(params[0], params[1], params[4], params[5])
	v = tw.getResultVector(s, params[2], params[3], weights, bias)
	return v

def basePrepare(product, numEpochs, alpha):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_"+product+".csv"
	#Create paths of training and testing reviews
	trainpath = "train_"+product+".csv"
	testpath = "test_"+product+".csv"

	#Read reviews
	df_train = pandas.read_csv(trainpath)
	df_test = pandas.read_csv(testpath)
	df = pandas.read_csv(allpath)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words
	trainX = np.array(process.formatBaseX(df_train, wordList))
	testX = np.array(process.formatBaseX(df_test, wordList))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY, numEpochs, alpha]

def baseDemo(product, numEpochs, alpha):
	params = basePrepare(product, numEpochs, alpha)
	tw.regression(params[0], params[1], params[2], params[3], params[4], params[5])

def baseVector(product, numEpochs, alpha):
	params = basePrepare(product, numEpochs, alpha)
	[s, weights, bias] = tw.getTrainingSession(params[0], params[1], params[4], params[5])
	v = tw.getResultVector(s, params[2], params[3], weights, bias)
	return v

def averagePrepare(product, numEpochs, alpha):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_"+product+".csv"
	#Create paths of training and testing reviews
	trainpath = "train_"+product+".csv"
	testpath = "test_"+product+".csv"

	#Read reviews
	df_train = pandas.read_csv(trainpath)
	df_test = pandas.read_csv(testpath)
	df = pandas.read_csv(allpath)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words
	trainX = np.array(process.formatAverageX(df_train, wordList))
	testX = np.array(process.formatAverageX(df_test, wordList))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY, numEpochs, alpha]

def averageDemo(product, numEpochs, alpha):
	params = averagePrepare(product, numEpochs, alpha)
	tw.regression(params[0], params[1], params[2], params[3], params[4], params[5])

def averageVector(product, numEpochs, alpha):
	params = averagePrepare(product, numEpochs, alpha)
	[s, weights, bias] = tw.getTrainingSession(params[0], params[1], params[4], params[5])
	v = tw.getResultVector(s, params[2], params[3], weights, bias)
	return v

def fusionPreapare(product, numEpochs, alpha, delta):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_"+product+".csv"
	#Create paths of training and testing reviews
	trainpath = "train_"+product+".csv"
	testpath = "test_"+product+".csv"

	#Read reviews
	df_train = pandas.read_csv(trainpath)
	df_test = pandas.read_csv(testpath)
	df = pandas.read_csv(allpath)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words using word2vec.txt
	trainX = np.array(process.formatFusionX(df_train, wordList, delta))
	testX = np.array(process.formatFusionX(df_test, wordList, delta))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))
	
	return [trainX, trainY, testX, testY, numEpochs, alpha]

def fusionDemo(product, numEpochs, alpha, delta):
	params = fusionPrepare(product, numEpochs, alpha, delta)
	tw.regression(params[0], params[1], params[2], params[3], params[4], params[5])

def fusionVector(product, numEpochs, alpha, delta):
	params = fusionPrepare(product, numEpochs, alpha, delta)
	[s, weights, bias] = tw.getTrainingSession(params[0], params[1], params[4], params[5])
	v = tw.getResultVector(s, params[2], params[3], weights, bias)
	return v

def compareModels(m1, m2, params):
	if m1 == "base":
		v1 = baseVector(params['product'], params['numEpochs'], params['alpha'])
	if m1 == "cluster":
		v1 = clusterVector(params['product'], params['numEpochs'], params['alpha'], params['delta'])
	if m1 == "average":
		v1 = averageVector(params['product'], params['numEpochs'], params['alpha'])
	if m1 == "fusion":
		v1 = fusionVector(params['product'], params['numEpochs'], params['alpha'], params['delta'])
	if m2 == "base":
		v2 = baseVector(params['product'], params['numEpochs'], params['alpha'])
	if m2 == "cluster":
		v2 = clusterVector(params['product'], params['numEpochs'], params['alpha'], params['delta'])
	if m2 == "average":
		v2 = averageVector(params['product'], params['numEpochs'], params['alpha'])
	if m2 == "fusion":
		v2 = fusionVector(params['product'], params['numEpochs'], params['alpha'], params['delta'])
	[n01, n10] = performance.countMissclassified(v1, v2)
	print("n01:",n01)
	print()
	print("n10:",n10)

def main(argv):
	params = {}
	model1 = argv[1]
	model2 = argv[2]
	params['product'] = argv[3]
	params['numEpochs'] = int(argv[4])
	params['alpha'] = float(argv[5])
	params['delta'] = float(argv[6])

	compareModels(model1, model2, params)
"""

	if model1 == "base":
		baseDemo(params['product'], params['numEpochs'], params['alpha'])
	elif model1 == "cluster":
		delta = float(argv[5])
		clusterDemo(product, numEpochs, alpha, delta)
	elif model1 == "average":
		averageDemo(product, numEpochs, alpha)
	elif model1 == "fusion":
		params['delta'] = float(argv[5])
		fusionDemo(product, numEpochs, alpha, delta)

"""
if __name__ == '__main__':
	main(sys.argv)
