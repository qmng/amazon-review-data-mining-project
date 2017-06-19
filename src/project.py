import pandas
import sys
import numpy as np
import tensorflow_wrapper as tw
import process
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

def clusterDemo(product, numEpochs, alpha, delta):
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
	
	#Tensor Flow regression
	tw.regression(trainX, trainY, testX, testY, numEpochs, alpha)

def baseDemo(product, numEpochs, alpha):
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

	#Tensor Flow regression
	tw.regression(trainX, trainY, testX, testY, numEpochs, alpha)

def main(argv):
	model = argv[1]
	product = argv[2]
	numEpochs = int(argv[3])
	alpha = float(argv[4])

	if model == "base":
		baseDemo(product, numEpochs, alpha)
	elif model == "cluster":
		delta = float(argv[5])
		clusterDemo(product, numEpochs, alpha, delta)

if __name__ == '__main__':
	main(sys.argv)
