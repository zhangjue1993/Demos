# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:16:38 2016

@author: zj
"""

# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import numpy

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, num):
	trainSize = int(num)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separatedata(data):
	separated = {}
	L={}
	for i in range(len(data)):
		vector = data[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	for j in separated:
		L[j]=len(separated[j])
	return [separated, L]

def caculateprior(data):
	separated = {}
	L={}
	for i in range(len(data)):
		vector = data[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	for j in separated:
		L[j]=len(separated[j])
	return L

def caculatecov(dataset,separated):
	cov={}
	copy=zip(*dataset)
	del copy[-1]
	for j in separated:
		a=zip(*separated[j]);
		del a[-1]
		cov[j]=numpy.corrcoef(a)
	mean=numpy.mean(copy,axis=0)
	return cov
def caculatemean(dataset):
	copy=dataset
	a=numpy.mean(copy,axis=0)
	mean=list(a);
	del mean[-1]
	return mean


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries



def calculateprobability(x, cov, L,mean):
	b=[]
	for i in range(len(x)-1):
		c=x[i]-mean[i]
		e=b.append(c)
	cov_in=numpy.linalg.inv(cov)
	cov_det=numpy.linalg.det(cov)
	E=numpy.array(b)
	exponent = math.exp(-numpy.dot(numpy.dot((E.T),cov_in),E)/2)
	p=1/(math.pow(2*math.pi,L/2))
	D=p*numpy.dot(math.sqrt(cov_det),exponent)
	#print numpy.dot(numpy.dot((E.T),cov_in),E)
	return D

def calculateClassprobabilities(cov, inputVector, p, num,L,mean):
	probabilities = {}
	for classValue in cov:
		probabilities[classValue] = 1
		c = cov[classValue]
		x = inputVector
		Prob=calculateprobability(x, c, L,mean)
		Prob=Prob*p[classValue]/num
		probabilities[classValue]=Prob
		print probabilities
	return probabilities


def predict(cov, inputVector,p,num,L,mean):
	probabilities = calculateClassprobabilities(cov, inputVector, p,num,L,mean)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(cov, testSet,p,num,L,mean):
	predictions = []
	for i in range(len(testSet)):
		result = predict(cov, testSet[i],p,num,L,mean)
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	data=[]
	filename='data.txt'
	with open(filename,'r') as f:
		for line in f:
			data.append(map(float,line.split()))
	print('Loaded data file {} with {} rows').format(filename, len(data))
	#caculate the interior probility
	separated, p=separatedata(data)
	
	num=255
	trainingSet, testSet = splitDataset(data, num)
	#print(zip(*testSet))

	Prior=caculateprior(trainingSet)
	# caculate the f
	cov=caculatecov(data,separated)
	mean=caculatemean(data)
	L=len(testSet[1])-1
	# prediction
	predictions = getPredictions(cov,testSet,Prior,num,L,mean)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

	print(L)

main()
 
