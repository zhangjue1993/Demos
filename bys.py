
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:24:14 2016

@author: zj
"""
import math
import random
def splitdata(data,num):
	trainSize = int(num)
	trainSet = []
	test = list(data)
	#test = list(data)
	while len(trainSet) < trainSize:
		index = random.randint(0,len(test)-1)
		trainSet.append(test.pop(index))
	return [trainSet, test]

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
	return L

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(data):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
	del summaries[-1]
	return summaries


def summarizeByClass(data):
	classdiv = separatedata(data)
	summaries = {}
	for classValue, instances in classdiv.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateprob(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateprob(x, mean, stdev)
	#probabilities*
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

data=[]
filename='data.txt'
with open(filename,'r') as f:
    for line in f:
        data.append(map(float,line.split()))
print('Loaded data file {} with {} rows')\
.format(filename, len(data))

trainset,testset = splitdata(data,255)
classdiv = separatedata(data)
summaries = summarizeByClass(trainset)
predictions = getPredictions(summaries, testset)
accuracy = getAccuracy(testset, predictions)
print('Accuracy: {0}%').format(accuracy)



