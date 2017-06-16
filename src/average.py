import numpy as np

def getAverageReviewVector(review, Z, index):
	Z = np.array(Z)

	avg = np.zeros(Z[0].shape)

	for key in review:
		for i in range(review[key]):
			avg += Z[index[key]]

	avg /= len(list(review.keys()))

	return avg

def main():
	review = {'a':1, 'b':1, 'c':1}
	Z = [[1,1], [2,2], [3,3]]
	index = {'a':0, 'b':1, 'c':2}
	print(getAverageReviewVector(review, Z, index))

if __name__ == '__main__':
	main()
