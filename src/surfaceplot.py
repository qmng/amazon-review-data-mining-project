import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

def findValue(df, col1, val1, col2, val2, col3):
	"""
	Return the value of 'col3' in dataframe 'df' where the value of column 'col1' is equal to 'val1' and the value of column 'col2' is equal to 'val2'.
	"""

	return df.loc[(df[col1] == val1) & (df[col2] == val2)][col3].iloc[0] # 0 is used to get the first row

def getXYZ(df, colX, colY, colZ):
	"""
	Computes meshgrid X,Y and matrix Z for surface plot.
	colX is the first column parameter, ex: learning rate.
	colY is the second column, ex: number of epochs
	"""

	# Get data of colX and colY in list format and remove duplicate values
	x = list(set(list(df[colX])))
	y = list(set(list(df[colY])))

	# Compute meshgrid using x and y
	X,Y = np.meshgrid(x,y)

	# Inizialise Z
	Z = np.zeros(X.shape)

	# For each combination (x,y) find the corresponding colZ value and add it to the matrix
	for i in range(len(x)):
		for j in range(len(y)):
			print(i)
			print(j)
			Z[i,j] = findValue(df, colX, x[i], colY, y[j], colZ)

	return [X,Y,Z]

def plot(X,Y,Z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X,Y,Z)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	plt.show()

def main():
	# Filtering test
	df = pd.read_csv('pandas_demo.csv')
#	print(findValue(df, 'col1', 0, 'col2', 1, 'col3')) # Print first row's value in 'col3'

	[X,Y,Z] = getXYZ(df, 'col1', 'col2', 'col3')
	print(X)
	print(Y)
	print(Z)


if __name__ == '__main__':
	main()
