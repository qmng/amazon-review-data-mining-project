import numpy as np

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
		print(temp)

		metawords.append(metaId)
		metadict[metaId] = temp
		metaId += 1

	return [metawords, metadict]

def createMetaWords(words, Z, index, delta):
	metaId = 0
	metawords = []
	metadict = {}


	while (words):
		w1 = words[0]
		temp = []
		temp.append(w1)
		v1 = np.array(Z[index[w1]])

		for w2 in words:
			v2 = np.array(Z[index[w2]])
			if np.linalg.norm(v1-v2) < delta and w1 != w2:
				 temp.append(w2)

		for w3 in temp:
			metadict[w3] = 'cluster'+str(metaId)
			words.remove(w3)

		metawords.append('cluster'+str(metaId))
		metaId += 1

	return [metawords, metadict]

def getMetaReview(review, metawords, metadict):
	metaReview = {}
	for i in range(len(metawords)):
		metaReview[metawords[i]] = 0

	for word in review:
		metaReview[metadict[word]] += review[word]

	return metaReview

def main():
	words = ['a', 'b', 'c', 'd', 'e', 'f']
	Z = [[1,2], [1,2], [2,2], [2,2], [3,3], [3,3]]
	index = {'a': 0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}
	delta = 10**(-15)

	[metawords, metadict] = createMetaWords(words, Z, index, delta)

	print(metawords)
	print(metadict)

	review = {'a':2, 'b':3, 'c':3, 'd':0, 'e':0, 'f':0}
	metaReview = getMetaReview(review, metawords, metadict)
	print(metaReview)

if __name__ == '__main__':
	main()
