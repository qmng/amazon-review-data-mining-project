import pandas
import glob
import os

df1 = pandas.read_csv("../../datasets/reviews_always.csv")
df2 = pandas.read_csv("../../datasets/reviews_gillette.csv")
df3 = pandas.read_csv("../../datasets/reviews_oral-b.csv")
df4 = pandas.read_csv("../../datasets/reviews_pantene.csv")
df5 = pandas.read_csv("../../datasets/reviews_tampax.csv")
train = df1.sample(frac=0.7)
test = df1.drop(train.index)
train.to_csv('train_always.csv')
test.to_csv('test_always.csv')

train = df2.sample(frac=0.7)
test = df2.drop(train.index)
train.to_csv('train_gillette.csv')
test.to_csv('test_gillette.csv')

train = df3.sample(frac=0.7)
test = df3.drop(train.index)
train.to_csv('train_oral-b.csv')
test.to_csv('test_oral-b.csv')

train = df4.sample(frac=0.7)
test = df4.drop(train.index)
train.to_csv('train_pantene.csv')
test.to_csv('test_pantene.csv')

train = df5.sample(frac=0.7)
test = df5.drop(train.index)
train.to_csv('train_tampax.csv')
test.to_csv('test_tampax.csv')
