import numpy as np
import project
import pandas

def metawordsTest():
	words = ['a', 'b', 'c', 'd', 'e', 'f']
	value = {'a': 1, 'b':1, 'c':100, 'd':100, 'e':1, 'f':100}

	metaId = 0

	metawords = []
	metadict = {}

	for w1 in words:
		temp = []
		temp.append(w1)
		for w2 in words:
			if value[w1]-value[w2] == 0 and w1 != w2:
				 temp.append(w2)

		for w3 in temp:
			words.remove(w3)

		metawords.append(metaId)
		metadict[metaId] = temp
		metaId += 1

	return [metawords, metadict]

def createMetaWords(words, Z, index, delta):
	"""
	Changes bag of words by combining words that are similar semantically.
	Words that are close have a close vectore representation.

	words: List of words in bag of words.
	Z: Matrix containing the vector representation of the words.
	index: Dictionary containing the index of each word in Z.
	delta: The maximum distance between two semantically close words.

	Returns the list of metawords and a dictionary where the keys are the words and the values are the corresponding metawords (mapping structure)
	"""

	metaId = 0
	metawords = []
	metadict = {}

	while (words):
		w1 = words[0]
		temp = []	# 'temp' will contain the words belonging to the same cluster
		temp.append(w1)	# Add w1 in the cluster
		v1 = np.array(Z[index[w1]]) # Find the word2vec vector of w1

		for w2 in words:	# Iterate through the words
			v2 = np.array(Z[index[w2]]) # Find the word2vec vector of w2
			if np.linalg.norm(v1-v2) < delta and w1 != w2: # If w1 is different than w2 and they are close enough in the word2vec space, then add w2 to the current cluster
				 temp.append(w2)

		for w3 in temp: # Iterate through words belonging to the current cluster
			metadict[w3] = 'cluster'+str(metaId)  # Add a new entry to the mapping structure for w3
			words.remove(w3) # Remove w3 from the list of words

		metawords.append('cluster'+str(metaId)) # Add the new cluster to the list of cluster
		metaId += 1

	return [metawords, metadict]

def getMetaReview(review, metawords, metadict):
	"""
	Computes the metareview of a review.


	review: The review in bag of words format.
	metawords: The list of metawords.
	metadict: A dictionary where keys are the normal words and values are the corresponding metawords
	"""

	metaReview = {}
	for i in range(len(metawords)): # Initialise the metaReview as a dictionary where keys are the metaword and the value are initially 0
		metaReview[metawords[i]] = 0

	for word in review: # Fill the metareview
		metaReview[metadict[word]] += review[word]

	return metaReview

def getMetaVect(review):
	return list(review.values())

def filterReviews(reviews, Zindex):
	"""
	Filter 'reviews' according to 'Zindex'.
	Words absent from 'Zindex' will be filtered out of reviews.
	"""

	res = []
	for review in reviews:
		temp = {}
		for key in review:
			if key in Zindex.keys():
				temp[key] = review[key]
		if temp:
			res.append(temp)
	return res

def convertMetaDict(metadict, metawords):
	"""
	Changes metadict format to be easier to process, especially in createMetaZ
	Metadict is in format {w1: mw1, w2: m1, w3: m2, w4: m2}
	Returned format is {m1: [w1, w2], m2: [w3, w4]}
	"""

	res = {}
	for mw in metawords:
		res[mw] =[]
		for key in metadict:
			if metadict[key] == mw:
				res[mw].append(key)

	return res

def createMetaZ(Z, index, metawords, metadict):
	"""
	Compute the metaZ which contains the word2vec representation of metawords.

	Returns metaZ and index (dictionary containing the metaword -> index in metaZ mapping)
	"""

	metaZ = []
	metaIndex = {}
	convertedDict = convertMetaDict(metadict, metawords)
	c = 0

	Z = np.array(Z)

	for mw in metawords:
		avg = np.zeros(Z[0].shape)
		uniqueWordCount = len(convertedDict[mw])
		for w in convertedDict[mw]:
			avg += Z[index[w]]
		metaZ.append(avg/uniqueWordCount)
		metaIndex[mw] = c
		c += 1

	return [metaZ, metaIndex]

def main():
	"""
	words = ['a', 'b', 'c', 'd', 'e', 'f']
	Z = [[1,2], [1,2], [2,2], [2,2], [3,3], [3,3]]
	index = {'a': 0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}
	delta = 10**(-15)
	df = pandas.read_csv("test_pantene.csv")

	[metawords, metadict] = formatClusterX(df)

	print(metawords)
	print(metadict)

	print(len(metawords))
	print(len(metadict))

	review = {'a':2, 'b':3, 'c':3, 'd':0, 'e':0, 'f':0}
	metaReview = getMetaReview(review, metawords, metadict)
	print(metaReview)
	"""

	words = ['a', 'b', 'c', 'd', 'e', 'f']
	Z = [[1,2], [1,2], [2,2], [2,2], [3,3], [3,3]]
	index = {'a': 0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}
	delta = 10**(-15)
	[metawords, metadict] = createMetaWords(words, Z, index, delta)

	print(metawords)
	print(metadict)

	[metaZ, metaIndex] = createMetaZ(Z, index, metawords, metadict)

	print(metaZ)
	print(metaIndex)

if __name__ == '__main__':
	main()
