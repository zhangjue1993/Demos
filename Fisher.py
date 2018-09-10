# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 17:19:24 2016

@author: zj
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def splitDataset(dataset, num):
	trainSize = int(num)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	trainSet=np.array(trainSet)
	copy=np.array(copy)
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

def ClassFeature(separated,l):
	mean={}
	var={}
	classmean=np.array(l)
	for i in separated:
		dat=np.array(separated[i])
		m,n=dat.shape
		data=dat[:,0:-1]
		mean[i]=np.mean(data,axis=0)
		var[i]=0
		for j in range(m-1):
			data[j]-=mean[i]
			c=data[j].reshape(l,-1)
			d=data[j].reshape(-1,l)
			v=np.dot(c,d)
			var[i]+=v
	np.array(mean)
	np.array(var)
	
	classmean=mean[1]-mean[2]
	return [classmean, var]

def CaculateW(var,mean,p):
	Sw=np.dot(p[1],var[1])+np.dot(p[2],var[2])
	Sw=np.linalg.inv(Sw)
	mean.shape=(-1,1)
	w=np.dot(Sw,mean)
	return w

def CaculateY(w,p,separated):
	y={}
	s=0
	w.shape=(1,-1)
	for i in separated:
		data=np.array(separated[i])
		y[i]=0
		m,n=data.shape
		for j in range(m):
			d=data[j,0:-1]
			k=np.dot(w,d)
			y[i]+=k/m
	for i in y:
		s=s+y[i]
	Y=s/2+np.log(p[1]/p[2])/(len(separated[1])+len(separated[2])-2)
	return Y
 
 
def Prediction(testSet,w,Y,p):
	w.shape=(1,-1)
	dat=testSet
	dat=np.arange(len(testSet))
	y=[]
	for i in range(len(testSet)):
		c=testSet[i,0:-1]
		y.append(np.dot(w,c))
		if (y[-1]>=Y):
			dat[i]=1
		else: dat[i]=2
	return [dat,y]

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i,-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 



def main():
	File="data.txt"
	data=np.loadtxt(File)
	p={1.0:0.6,2.0:0.4}
	num=72
	trainingSet, testSet = splitDataset(data, num)
	if num==72:
		testSet=trainingSet
	m,n=trainingSet.shape
	separated,L=separatedata(trainingSet)
	average,var=ClassFeature(separated,n-1)
	w=CaculateW(var,average,p)
	Y0=CaculateY(w,p,separated)
	Predictions,y=Prediction(testSet,w,Y0,p)
	Accuracy=getAccuracy(testSet, Predictions)
	print w
	print('Accuracy: {0}%').format(Accuracy)

	separated,L=separatedata(testSet)

	fig = plt.figure()
	ax=plt.subplot(111,projection='3d')
	
	data=np.array(separated[1])
	x=data[:,0]
	y=data[:,1]
	z=data[:,2]
	ax.scatter(x, y, z,c='y')
	data=np.array(separated[2])
	x=data[:,0]
	y=data[:,1]
	z=data[:,2]
	ax.scatter(x, y, z,c='r')

	X=np.arange(0, 4, 0.25)
	Y=np.arange(0, 4, 0.25)
	X, Y = np.meshgrid(X, Y)
	Z=(Y0-(np.dot(w[0,0],X)+np.dot(w[0,1],Y)))/w[0,2]
	#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
	##ax=plt.subplot(111,projection='3d')
	##	if y[i]==1:
		
main()