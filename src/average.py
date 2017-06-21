import numpy as np

def getAverageReviewVector(review, Z, index):
	"""
	Converts a review in bag of words representation to average vector representation.
	Sums each word's vector representation then divide by word count not unique word count, ex:
	{'a':1, 'b':2, 'c':1} -> word count = 4, unique word count = 3
	"""

	Z = np.array(Z)

	avg = np.zeros(Z[0].shape)
	wordsNumber = 0

	for key in review:
		avg += review[key]*Z[index[key]]
		wordsNumber += review[key] # compute word count

	avg /= wordsNumber

	return avg

def main():
	review = {'a':2, 'b':1, 'c':1}
	Z = [[1,1], [2,2], [3,3]]
	index = {'a':0, 'b':1, 'c':2}
	print(getAverageReviewVector(review, Z, index))

if __name__ == '__main__':
	main()
