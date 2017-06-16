from collections import Counter

def sum(l):
	cnt = Counter()
	for d in l:
		cnt += Counter(d)
	return cnt

d1 = {'a':0, 'b':1, 'c':2}
d2 = {'a':2, 'b':2, 'c':0}
d3 = {'a':0, 'c':3, 'b':0}

l = []
l.append(d1)
l.append(d2)
l.append(d3)

print(dict(sum(l)))
