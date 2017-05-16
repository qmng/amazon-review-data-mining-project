import pandas
import sys
import re
import copy

def removeApostrophe(reviews):
	for i in range(len(reviews)):
		reviews[i] = str(reviews[i]).replace("\\'", "")
	return reviews

def removeWord(review, word):
	return re.sub(r"\b{}\b".format(word), "", review)

def removeWords(review, words):
	for w in words:
		review = removeWord(review, w)
	return review

def removeWordsFromList(reviews, words):
	for i in range(len(reviews)):
		reviews[i] = removeWords(reviews[i], words)
	return reviews

def removeNonAlphanumeric(reviews):
	for i in range(len(reviews)):
		reviews[i] = re.sub('[^A-Za-z0-9\s]+', ' ', reviews[i])
	return reviews

def lowercase(reviews):
	for i in range(len(reviews)):
		reviews[i] = reviews[i].lower()
	return reviews

def getWordList(reviews):
	wordList = []
	for i in range(len(reviews)):
		wordList.extend(reviews[i].split())
	return wordList

def initDictionary(wordList):
	d = {}
	for i in range(len(wordList)):
		d[wordList[i]] = 0
	return d

def transformReviews(reviews, d):
	copyDict = copy.copy(d)
	for i in range(len(reviews)):
		for word in reviews[i].split():
			d[word] += 1
		reviews[i] = d
		d = copyDict
	return reviews

def printNonZero(d):
	for key in d:
		if d[key] != 0:
			print(key, d[key])

def createZ(d, filePath):
	Z = []
	f = open(filePath, "r")
	for line in f:
		l = line.split()
		if l[0] in d:
			Z.append([float(v) for v in l[1:len(l)]])

	return Z

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
	# Stem the reviews
	reviews = removeWordsFromList(reviews, stopWords)
	wordList = getWordList(reviews)
	d = initDictionary(wordList)

	reviews = transformReviews(reviews, d)

	rating = list(df1['user_rating'])

	print(d['line'])

	print('flirty' in d)

	Z = createZ(d, "../../word2vec/word2vec.txt")
	print(len(Z))

if __name__ == '__main__':
	main(sys.argv)
