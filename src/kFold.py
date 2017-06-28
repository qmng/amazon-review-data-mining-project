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

def clusterPrepare(i, delta):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_all.csv"
	#Create paths of training and testing reviews
	testpath="kFold/test_all_"+str(i)+".csv"

	#Read reviews
	df = pandas.read_csv(allpath)
	df_test = pandas.read_csv(testpath)
	df_train = df.drop(df_test.index)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words using word2vec.txt
	trainX = np.array(process.formatClusterX(df_train, wordList, delta))
	testX = np.array(process.formatClusterX(df_test, wordList, delta))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY]

def clusterDemo(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultAccuracy(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def clusterVector(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultVector(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def basePrepare(i):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_all.csv"
	#Create paths of training and testing reviews
	testpath="kFold/test_all_"+str(i)+".csv"

	#Read reviews
	df = pandas.read_csv(allpath)
	df_test = pandas.read_csv(testpath)
	df_train = df.drop(df_test.index)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words
	trainX = np.array(process.formatBaseX(df_train, wordList))
	testX = np.array(process.formatBaseX(df_test, wordList))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY]

def baseDemo(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultAccuracy(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def baseVector(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultVector(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def averagePrepare(i):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_all.csv"
	#Create paths of training and testing reviews
	testpath="kFold/test_all_"+str(i)+".csv"

	#Read reviews
	df = pandas.read_csv(allpath)
	df_test = pandas.read_csv(testpath)
	df_train = df.drop(df_test.index)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words
	trainX = np.array(process.formatAverageX(df_train, wordList))
	testX = np.array(process.formatAverageX(df_test, wordList))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY]

def averageDemo(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultAccuracy(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def averageVector(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultVector(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def fusionPrepare(i, delta):
	#Create path of all the reviews
	allpath = "../../datasets/reviews_all.csv"
	#Create paths of training and testing reviews
	testpath="kFold/test_all_"+str(i)+".csv"

	#Read reviews
	df = pandas.read_csv(allpath)
	df_test = pandas.read_csv(testpath)
	df_train = df.drop(df_test.index)

	#Prepare the wordList for the product
	reviews = process.processReviews(df)
	wordList = process.preprocessWordlist(reviews)

	#Create the vectors of the bag of words using word2vec.txt
	trainX = np.array(process.formatFusionX(df_train, wordList, delta))
	testX = np.array(process.formatFusionX(df_test, wordList, delta))

	#Create the vectors of the evaluations
	trainY = np.array(process.formatY(df_train))
	testY = np.array(process.formatY(df_test))

	return [trainX, trainY, testX, testY]

def fusionDemo(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultAccuracy(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def fusionVector(numEpochs, alpha, lossFunc, setup):
	v = tw.getResultVector(setup[0], setup[1], setup[2], setup[3], numEpochs, alpha, setup[4], setup[5], lossFunc)
	return v

def setupModel(m, prod, delta):
	if m == "base":
		p = basePrepare(prod)
	if m == "cluster":
		p =  clusterPrepare(prod, delta)
	if m == "average":
		p =  averagePrepare(prod)
	if m == "fusion":
		p =  fusionPrepare(prod, delta)
	weights = tw.getInitWeights(p[0], p[1])
	bias = tw.getInitBias(p[1])
	p.append(weights)
	p.append(bias)
	# Returns array with: [TrainX, TrainY, TestX, TestY, weights, bias]
	return p

def compareModels(m1, m2, params, setup1, setup2, i):
	"""
	Calculates success rate and mcNemar test with models m1 and m2
	For reviews_all.csv k fold cross validation.
	Stores results in Results/kFold/
	"""

	if m1 == "base":
		v1 = baseVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup1)
	if m1 == "cluster":
		v1 = clusterVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup1)
	if m1 == "average":
		v1 = averageVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup1)
	if m1 == "fusion":
		v1 = fusionVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup1)

	a1 = tw.getResultAccuracy3(v1)

	if m2 == "base":
		v2 = baseVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup2)
	if m2 == "cluster":
		v2 = clusterVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup2)
	if m2 == "average":
		v2 = averageVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup2)
	if m2 == "fusion":
		v2 = fusionVector(params['numEpochs'], params['alpha'], params['lossFunc'], setup2)
	[n01, n10] = performance.countMissclassified(v1, v2)

	a2 = tw.getResultAccuracy3(v2)

	resNemar = [params['lossFunc'], m1, m2, n01, n10, params['alpha'], params['numEpochs'], params['delta'], i]
	resSuccess = [params['lossFunc'], m1, m2, a1, a2, params['alpha'], params['numEpochs'], params['delta'], i]
	dfNemar = pandas.DataFrame([resNemar])
	pathNemar = "Results/kFold/mcnemar_all_"+str(i)+".csv"
	pathSuccess = "Results/kFold/success_all_"+str(i)+".csv"
	if os.path.isfile(pathNemar):
		dfNemar.to_csv(open(pathNemar, 'a', encoding='utf-8-sig'), index=False, header=False, encoding='utf-8-sig')
		print("printed mcnemar.csv")
	else:
		dfNemar.to_csv(open(pathNemar, 'a', encoding='utf-8-sig'), index=False, header=['lossFunc', 'nom1', 'nom2', 'n01', 'n10', 'alpha', 'numEpochs', 'delta', 'k'], encoding='utf-8-sig')
		print("printed mcnemar.csv")
	dfSuccess = pandas.DataFrame([resSuccess])
	if os.path.isfile(pathSuccess):
		dfSuccess.to_csv(open(pathSuccess, 'a', encoding='utf-8-sig'), index=False, header=False, encoding='utf-8-sig')
		print("printed success.csv")
	else:
		dfSuccess.to_csv(open(pathSuccess, 'a', encoding='utf-8-sig'), index=False, header=['lossFunc', 'nom1', 'nom2', 'success1', 'success2', 'alpha', 'numEpochs', 'delta', 'k'], encoding='utf-8-sig')
		print("printed success.csv")

def main(argv):
	params = {}
	model1 = argv[1]
	model2 = argv[2]
	params['lossFunc'] = argv[3]
	params['delta'] = 40
	params['alpha'] = 0.1
	params['numEpochs'] = 10000

	for i in range(0,10):
		setup1 = setupModel(model1, i, params['delta'])
		setup2 = setupModel(model2, i, params['delta'])
		compareModels(model1, model2, params, setup1, setup2)
"""
	for delta in range(10, 41, 10):
		params['delta'] = float(delta)
		setup1 = setupModel(model1, params['product'], delta)
		setup2 = setupModel(model2, params['product'], delta)
		for i in range(0, 2):
			params['alpha'] = float(10**(-i))
			for nepochs in range(10000, 100001, 10000):
				params['numEpochs'] = nepochs
				compareModels(model1, model2, params, setup1, setup2)
"""
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

