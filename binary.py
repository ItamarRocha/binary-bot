#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:33:29 2020

@author: itamar
"""

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



SEQ_LEN = 30  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 2  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 5
BATCH_SIZE = 8
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop('future',1)
    
    dataset = df
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace = True)
            df[col] = preprocessing.scale(df[col].values)
    
    df.dropna(inplace = True)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)            
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq , target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys),len(sells))

    buys = buys[:lower]
    sells = sells[:lower]    
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq,target in sequential_data:
        X.append(seq)
        y.append(target)
        
    
    return np.array(X), y
    
ratio = 'EURUSD'
dataset = f'binary_options_data/{ratio}.csv'  # get the full path to the file.
df = pd.read_csv(dataset).drop(columns = {'from','to','id'})  # read in specific file

df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
df.dropna(inplace=True)

df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
df['volume'] = df["volume"].shift(-FUTURE_PERIOD_PREDICT)
df['open'] = df["open"].shift(-FUTURE_PERIOD_PREDICT)
df['min'] = df["min"].shift(-FUTURE_PERIOD_PREDICT)
df['max'] = df["max"].shift(-FUTURE_PERIOD_PREDICT)

df['target'] = list(map(classify,df["close"],df["future"]))

times = sorted(df.index.values)
last_5pct = times[-int(0.1*len(times))]

validation_main_df = df[(df.index >= last_5pct)]
df = df[(df.index < last_5pct)]

train_x, train_y = preprocess_df(df)

validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont Buys: {validation_y.count(0)}, buys : {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1:]), return_sequences = True))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
"""
model.add(LSTM(128, return_sequences = True))
model.add(Dropout(0.1))
model.add(BatchNormalization())
"""
model.add(LSTM(128))
#model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.15)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

tensorboard = TensorBoard(log_dir = f'logs/{NAME}')

filepath = "RNN_FINAL-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS,validation_data = (validation_x,validation_y),callbacks=[tensorboard,checkpoint])

import matplotlib.pyplot as pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()