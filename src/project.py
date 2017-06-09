import pandas
import sys
import re
import copy
import subprocess
from collections import Counter

def removeApostrophe(reviews):
	res = []
	for i in range(len(reviews)):
		res.append(str(reviews[i]).replace("\\'", ""))
		#reviews[i] = str(reviews[i]).replace("\\'", "")
	return res

def removeWord(review, word):
	# Removes the word 'word' from the review 'review'
	return re.sub(r"\b{}\b".format(word), "", review)

def removeWords(review, words):
	# Removes the words contained in the list 'words' from the review 'review'
	for w in words:
		review = removeWord(review, w)
	return review

def removeWordsFromList(reviews, words):
	res = []
	# Removes the words in the list 'words' from the list of review 'reviews'
	for i in range(len(reviews)):
		res.append(removeWords(reviews[i], words))
		#reviews[i] = removeWords(reviews[i], words)
	return res

def removeNonAlphanumeric(reviews):
	# Removes non alpha numeric characters from the list of reviews 'reviews'
	res = []
	for i in range(len(reviews)):
		res.append(re.sub('[^A-Za-z0-9\s]+', ' ', reviews[i]))
		#reviews[i] = re.sub('[^A-Za-z0-9\s]+', ' ', reviews[i])
	return res

def stemReviews(reviews, path):
	res = []
	for i in range(len(reviews)):
		res.append(stemReview(reviews[i], path))
		#reviews[i] = stemReview(reviews[i], path)
	return res

def stemReview(review, path):
	t = subprocess.Popen(["perl", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	stemmed = t.communicate(input=review)
	return stemmed[0].strip()

def lowercase(reviews):
	# Converts the list of review 'reviews' to lower case
	res = []
	for i in range(len(reviews)):
		res.append(reviews[i].lower())
		#reviews[i] = reviews[i].lower()
	return res

def getWordList(reviews):
	# Returns a list containing the words in the list of review 'reviews'
	wordList = []
	for i in range(len(reviews)):
		wordList.extend(reviews[i].split())
	return wordList

def initDictionary(wordList):
	# Creates the initial dictionary where the keys are the words in 'wordList'
	# The dictionary contains the keys but each value is 0
	d = {}
	for i in range(len(wordList)):
		d[wordList[i]] = 0
	return d

def transformReviews(reviews, d):
	# Transforms each review into its bag-of-words representation
	copyDict = d.copy()
	res = []
	for i in range(len(reviews)):
		for word in reviews[i].split():
			d[word] += 1
		res.append(d)
		#reviews[i] = d
		d = copyDict.copy()
	return res

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

def createZ(d, filePath):
	# Creates matrix Z containg the vector representation of the words in the dictionary
	# Also returns the index to get the correspondig vector of a word
	# Example:
	# index['amazon'] will return the index of the vector of 'amazon' in Z
	# Z[index['amazon']] will return the vector of 'amazon'
	Z = []
	i = 0
	index = {}
	keys = list(d)
	f = open(filePath, "r")
	for line in f:
		l = line.split() # get the list of the current line, l[0] is the word, l[1:len(l)] is the corresponding vector
		if l[0] in d: # if the word is in the dictionary then we add it to Z
			Z.append([float(v) for v in l[1:len(l)]])
			index[l[0]] = i
			i += 1

	return [Z, index]

def filterRating(X, Y, r):
	listIndex = []
	filteredX = []
	for n in range(len(Y)):
		if Y[n] == r:
			listIndex.append(n)
	for i in listIndex:
		filteredX.append(X[i])
	return filteredX

def sumListDictionary(X):
	temp = Counter()
	test = 0
	for n in X:
		test = test + n['failed']
		temp += Counter(n)
	return dict(temp)

def main(argv):
	#df1 = pandas.read_csv("../../datasets/reviews_always.csv")
	#df1 = pandas.read_csv("../../datasets/reviews_gillette.csv")
	df1 = pandas.read_csv("../../datasets/reviews_oral-b.csv")
	#df1 = pandas.read_csv("../../datasets/reviews_pantene.csv")
	#df1 = pandas.read_csv("../../datasets/reviews_tampax.csv")

	stopWords = [line.rstrip('\n') for line in open('../../stop_words.txt')]
	#specialChar = ['\.', '\,', '\?', '\!', '\(', '\)', '\:', '\-', "\\'"]

	reviews = list(df1['review'])
	reviews = removeApostrophe(reviews)
	reviews = removeNonAlphanumeric(reviews)
	reviews = lowercase(reviews)
	stemReviews(reviews, "./stemmer.pl")
	reviews = removeWordsFromList(reviews, stopWords)
	wordList = getWordList(reviews)
	d = initDictionary(wordList)

	X = transformReviews(reviews, d)
	Y = list(df1['user_rating'])
	rating1 = filterRating(X, Y, 5)
	rating2 = filterRating(X, Y, 5)
	rating3 = filterRating(X, Y, 5)
	rating4 = filterRating(X, Y, 5)
	rating5 = filterRating(X, Y, 5)
#	[Z, index] = createZ(d, "../../word2vec/word2vec.txt")

if __name__ == '__main__':
	main(sys.argv)
