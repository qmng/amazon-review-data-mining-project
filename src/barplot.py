import numpy as np
import matplotlib.pyplot as plt

def plot(x,y):
	plt.bar(range(len(y)), y)
	plt.xticks(range(len(y)), x)
	plt.show()

def main():
	x = ['Base', 'Cluster', 'Average', 'Fusion']
	y = [0.6, 0.7, 0.8, 0.9]
	plot(x,y)

if __name__ == '__main__':
	main()
