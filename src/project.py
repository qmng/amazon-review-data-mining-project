import pandas
import sys
import numpy as np
import tensorflow_wrapper as tw
import process
import performance
import os.path
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

def clusterDemo(product, numEpochs, alpha, delta, lossFunc):
	params = clusterPrepare(product, numEpochs, alpha, delta)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultAccuracy(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def clusterVector(product, numEpochs, alpha, delta, lossFunc):
	params = clusterPrepare(product, numEpochs, alpha, delta)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultVector(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
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

def baseDemo(product, numEpochs, alpha, lossFunc):
	params = basePrepare(product, numEpochs, alpha)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultAccuracy(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def baseVector(product, numEpochs, alpha, lossFunc):
	params = basePrepare(product, numEpochs, alpha)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultVector(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
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

def averageDemo(product, numEpochs, alpha, lossFunc):
	params = averagePrepare(product, numEpochs, alpha)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultAccuracy(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def averageVector(product, numEpochs, alpha, lossFunc):
	params = averagePrepare(product, numEpochs, alpha)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultVector(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def fusionPrepare(product, numEpochs, alpha, delta):
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

def fusionDemo(product, numEpochs, alpha, delta, lossFunc):
	params = fusionPrepare(product, numEpochs, alpha, delta)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultAccuracy(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def fusionVector(product, numEpochs, alpha, delta, lossFunc):
	params = fusionPrepare(product, numEpochs, alpha, delta)
	weights = tw.getInitWeights(params[0], params[1])
	bias = tw.getInitBias(params[1])
#	s = tw.getTrainingSession(params[0], params[1], params[4], params[5], weights, bias, lossFunc)
	v = tw.getResultVector(params[0], params[1], params[2], params[3], params[4], params[5], weights, bias, lossFunc)
	return v

def compareModels(m1, m2, params):
	if m1 == "base":
		a1 = baseDemo(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
		v1 = baseVector(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
	if m1 == "cluster":
		a1 = clusterDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
		v1 = clusterVector(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
	if m1 == "average":
		a1 = averageDemo(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
		v1 = averageVector(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
	if m1 == "fusion":
		a1 = fusionDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
		v1 = fusionVector(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
	if m2 == "base":
		a2 = baseDemo(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
		v2 = baseVector(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
	if m2 == "cluster":
		a2 = clusterDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
		v2 = clusterVector(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
	if m2 == "average":
		a2 = averageDemo(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
		v2 = averageVector(params['product'], params['numEpochs'], params['alpha'], params['lossFunc'])
	if m2 == "fusion":
		a2 = fusionDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
		v2 = fusionVector(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc'])
	[n01, n10] = performance.countMissclassified(v1, v2)
	resNemar = [params['lossFunc'], m1, m2, n01, n10, params['alpha'], params['numEpochs'], params['delta']]
	resSuccess = [params['lossFunc'], m1, m2, a1, a2, params['alpha'], params['numEpochs'], params['delta']]
	dfNemar = pandas.DataFrame([resNemar])
	pathNemar = "Results/"+params['product']+"/mcnemar_"+params['product']+".csv"
	pathSuccess = "Results/"+params['product']+"/success_"+params['product']+".csv"
	if os.path.isfile(pathNemar):
		dfNemar.to_csv(open(pathNemar, 'a', encoding='utf-8-sig'), index=False, header=False, encoding='utf-8-sig')
	else:
		dfNemar.to_csv(open(pathNemar, 'a', encoding='utf-8-sig'), index=False, header=['lossFunc', 'nom1', 'nom2', 'n01', 'n10', 'alpha', 'numEpochs', 'delta'], encoding='utf-8-sig')
	dfSuccess = pandas.DataFrame([resSuccess])
	if os.path.isfile(pathSuccess):
		dfSuccess.to_csv(open(pathSuccess, 'a', encoding='utf-8-sig'), index=False, header=False, encoding='utf-8-sig')
	else:
		dfSuccess.to_csv(open(pathSuccess, 'a', encoding='utf-8-sig'), index=False, header=['lossFunc', 'nom1', 'nom2', 'success1', 'success2', 'alpha', 'numEpochs', 'delta'], encoding='utf-8-sig')

def main(argv):
	params = {}
	model1 = argv[1]
	model2 = argv[2]
	params['product'] = argv[3]
	params['lossFunc'] = argv[4]

	for nepochs in range(500, 501, 500):
		params['numEpochs'] = nepochs
		for i in range(0, 10):
			params['alpha'] = float(10**(-i))
			for delta in range(-2, 3):
				params['delta'] = float(10**delta)
				compareModels(model1, model2, params)
#	params['numEpochs'] = int(argv[4])
#	params['alpha'] = float(argv[5])
#	params['delta'] = float(argv[6])

#	print(clusterDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'], params['lossFunc']))
#	compareModels(model1, model2, params)
"""

	if model1 == "base":
		baseDemo(params['product'], params['numEpochs'], params['alpha'])
	elif model1 == "cluster":
		delta = float(argv[5])
		clusterDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'])
	elif model1 == "average":
		averageDemo(params['product'], params['numEpochs'], params['alpha'])
	elif model1 == "fusion":
		params['delta'] = float(argv[5])
		fusionDemo(params['product'], params['numEpochs'], params['alpha'], params['delta'])
"""

if __name__ == '__main__':
	main(sys.argv)
