import pandas
import glob
import os
import numpy

#Creates test.csv for oral-b's kFold cross validation
df = pandas.read_csv("../../datasets/reviews_oral-b.csv")
df = df.sample(frac=1)
k = 10
splitLen = round(len(df)/k)
tests = []
tests.append(df[:splitLen])
for k in range(1,k-1):
	tempLen1 = k*splitLen
	tempLen2 = (k+1)*splitLen
	tests.append(df[tempLen1:tempLen2])
tests.append(df[tempLen2:])
for n in range(len(tests)):
	path = 'kFold/test_oral-b_'+str(n)+'.csv'
	tests[n].to_csv(path)
