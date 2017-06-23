import sys

def countMissclassified(test1, test2):
	#n01: test1 correctly classified but test2 missclassified
	n01 = 0
	#n10: test1 missclassified but test2 correctly classified
	n10 = 0
	for i in range(len(test1)):
		if test1[i]:
			if not(test2[i]):
				n01 = n01 + 1
		elif not(test1[i]):
			if test2[i]:
				n10 = n10 + 1
	return [n01, n10]

def main(argv):
	t1 = [True, True, False, False, True]
	t2 = [False, True, False, True, False]

	[n01, n10] = countMissclassified(t1, t2)
	print("n01:", n01)
	print()
	print("n10:",n10)

if __name__ == '__main__':
	main(sys.argv)
