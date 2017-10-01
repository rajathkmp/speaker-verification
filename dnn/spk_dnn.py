from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, log_loss

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import keras.backend as K

import numpy as np
import htkmfc as htk
import sys
import os

np.random.seed(2570)

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

def realData(data, SPKlabel ,KWlabel):
	tempIndex = np.where( KWlabel == 1)[0]
	dataReal = []
	spklabelReal = []
	for i in range(len(data)):
		if i in tempIndex:
			dataReal.append(data[i])
			spklabelReal.append(SPKlabel[i])
	return (np.array(dataReal), np.array(spklabelReal))

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
	pathToData = '/home/rajathk/spkVer/code/dataGen/spkData/'+sys.argv[1]+'_'+str(windowSize)+'/'+sys.argv[1]+'_train_spk.htk'

	X_train, Y_train_spk, Y_train_kw, Y_train_utt = loadData(pathToData)
	Y_train_spk = correctLabel(Y_train_spk)

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)

	joblib.dump(scaler, 'scaler.save')

	X_train, Y_train_spk = realData(X_train, Y_train_spk, Y_train_kw)

	X_train, X_dev, Y_train_spk, Y_dev_spk = train_test_split(X_train, Y_train_spk, test_size = 0.20)

	spkLen = len(set(Y_train_spk))

	Y_train_spk = np_utils.to_categorical(Y_train_spk, spkLen)
	Y_dev_spk = np_utils.to_categorical(Y_dev_spk, spkLen)

	model = Sequential()
	model.add(Dense(256, input_shape=(X_train.shape[1], )))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(256))
	
	modelInput = Input(shape=(X_train.shape[1] ,  ))
	features = model(modelInput)

	spkModel = Model(inputs = modelInput, outputs=features)

	model1 = Activation('relu')(features)
	model1 = Dropout(0.3)(model1)
	model1 = Dense(spkLen, activation='softmax')(model1)

	spk = Model(inputs=modelInput, outputs=model1)

	sgd = SGD( lr = 0.02)
	early_stopping = EarlyStopping(monitor='val_loss', patience=4)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)
	csv_logger = CSVLogger('training.log')
	spk.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])
	spk.fit(X_train, Y_train_spk, batch_size = 128, epochs = 150, verbose = 1, validation_data = (X_dev, Y_dev_spk), callbacks = [early_stopping, reduce_lr, csv_logger])
	spk.save('spk.h5')
	spkModel.save('onlySPK.h5')
