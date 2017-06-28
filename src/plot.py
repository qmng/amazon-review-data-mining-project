import numpy as np
import matplotlib.pyplot as plt

def main():
	delta = [0,5,10,15,20,25,30,35,40,45,50]
	s1 = [2800, 2627, 2381, 2093, 1287, 455, 71, 3, 1, 1, 1]
	s2 = [10100, 9200, 8202, 6688, 3795, 1169, 186, 12, 1, 1, 1]

	plt.xlabel('delta')
	plt.ylabel('average cluster size')
	plt.plot(delta, s1)

if __name__ == '__main__':
	main()
