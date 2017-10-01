import numpy as np
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from itertools import combinations
import htkmfc as htk
import logging
import random
import time
import sys
import os

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.spatial.distance import cosine

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.models import load_model

def loadData(inputData):

	featsReader = htk.open(inputData)
	trainData = featsReader.getall()

	np.random.shuffle(trainData)

	yUtt = trainData[:, -1]
	trainData = np.delete(trainData, -1, 1)
	
	ySpkTrain = trainData[:, -1]
	trainData = np.delete(trainData, -1, 1)
	
	yKwTrain = trainData[:, -1]
	xTrain = np.delete(trainData, -1, 1)
	
	del trainData

	return (xTrain, ySpkTrain.astype(int), yKwTrain.astype(int) ,yUtt.astype(int))

def correctLabel(yTrain):
	a = list(set(yTrain))
	correctA = [ i  for i in range(len(a)) ]
	
	Y_trainFin = []
	for i in yTrain:
		Y_trainFin.append(correctA[a.index(i)])
		
	return np.array(Y_trainFin)

def cnn_reshape(X_list, windowSize):

	Y_list = []
	for i in X_list:
		j = i.reshape(windowSize,32)
		Y_list.append(j)
	Y_list = np.array(Y_list)
	Z_list = Y_list[:, np.newaxis, :, :]
	return(Z_list)

def realData(data, SPKlabel ,KWlabel, uttLabel):
	tempIndex = np.where( KWlabel == 1)[0]

	dataReal = []
	spklabelReal = []
	utt = []
	for i in range(len(data)):
		if i in tempIndex:
			dataReal.append(data[i])
			spklabelReal.append(SPKlabel[i])
			utt.append(uttLabel[i])
	return (np.array(dataReal), np.array(spklabelReal), np.array(utt))

def normVec(vec):
	return(vec/np.linalg.norm(vec))


def queryWS(nameKW):
	a = {
	'government': 75,
	'company': 71,
	'hundred': 59,
	'nineteen': 79,
	'thousand': 77,
	'morning': 69,
	'business': 81
	}

	return a[nameKW]

if __name__ == "__main__":

	windowSize = queryWS(sys.argv[1])
	pathToData = '/home/rajathk/spkVer/code/dataGen/spkData/'+sys.argv[1]+'_'+str(windowSize)+'/'+sys.argv[1]+'_test_spk.htk'

	X_train, Y_train_spk, Y_train_kw, Y_train_utt = loadData(pathToData)
	Y_train_spk = correctLabel(Y_train_spk)

	scaler = joblib.load('scaler.save')
	X_train = scaler.transform(X_train)

	X_train, Y_train, Y_train_utt = realData(X_train, Y_train_spk, Y_train_kw, Y_train_utt)	
	X_train = cnn_reshape(X_train, windowSize)

	model = load_model('onlySPK.h5')

	X_train = model.predict(X_train)
	X_train = np.array([ normVec(i) for i in X_train ])

	xTrain = []
	yTrain = []

	for i in list(set(Y_train_utt)):

		tempIndex = np.where(Y_train_utt == i)[0]
		dataTemp = []
		labelTemp = []

		for j in tempIndex:
			dataTemp.append(X_train[j])
			labelTemp.append(Y_train[j])

		if len(set(labelTemp)) != 1:
			print(i)
			sys.exit()
			break
		xTrain.append(np.average(np.array(dataTemp), axis = 0))
		yTrain.append(np.mean(labelTemp).astype(int))

	diffSpkD = []

	for i in range(len(set(yTrain))):
		tempIndex = np.where(np.array(yTrain) == i)[0]
		diffSpkD.append(combinations(tempIndex, 3))

	TrueLabel = []
	FalseLabel = []

	for i in range(len(diffSpkD)):
		print('{} of {}'.format(i, len(diffSpkD) - 1 ))
		for j in diffSpkD[i]:
			spkDtemp = []
			for eachOne in j:
				spkDtemp.append(xTrain[eachOne])
			spkD = np.average(spkDtemp, axis=0)

			for m in range(len(yTrain)):
				if m not in list(j):
					tempData = xTrain[m]
					cosineDist = 1 - cosine(spkD, tempData)

					if i == yTrain[m]:
						TrueLabel.append(cosineDist)
					else:
						FalseLabel.append(cosineDist)



	groundTr = [0]* len(FalseLabel) + [1]* len(TrueLabel)
	posteriors = FalseLabel + TrueLabel

	print('FalseLen: {} , TrueLen: {}'.format(len(FalseLabel), len(TrueLabel)))
	auc = roc_auc_score(groundTr, posteriors)
	print('area under curve: ', auc*100)

	far, tar, thr = roc_curve(groundTr, posteriors)
	far = far*100
	frr = (1 - tar)*100

	minDiff = min([ abs(far[i] - frr[i]) for i in range(len(far)) ])

	for i in range(len(far)):
		if abs(far[i] - frr[i]) == minDiff:
			eer = (far[i]+frr[i])/2
			print('EER: ', eer)
			break



