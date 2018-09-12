#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keras implementation of Dilated Causal Convolutional Neural Network for Time 
Series Predictions based on the following sources:

[1] A. van den Oord et al., “Wavenet: A generative model for raw audio,” arXiv 
    preprint arXiv:1609.03499, 2016.

[2] A. Borovykh, S. Bohte, and C. W. Oosterlee, “Conditional Time Series 
    Forecasting with Convolutional Neural Networks,” arXiv:1703.04691 [stat], 
    Mar. 2017.

Initial 1D convolutional code structure based on:
https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee

Author: Krist Papadopoulos
V0 Date: March 31, 2018
V1 Data: September 12, 2018
         - updated Keras merge function to Add for Keras 2.2.2
        
         tensorflow==1.10.1
         Keras==2.2.2
         numpy==1.14.5
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv1D, Input, Add, Activation, Dropout

from keras.models import Sequential, Model

from keras.regularizers import l2

from keras.initializers import TruncatedNormal

from keras.layers.advanced_activations import LeakyReLU, ELU

from keras import optimizers


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    def f(input_):
        
        residual =    input_
        
        layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                      dilation_rate=dilation, 
                      activation='linear', padding='causal', use_bias=False,
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(input_)
                    
        layer_out =   Activation('selu')(layer_out)
        
        skip_out =    Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
        
        network_in =  Conv1D(1,1, activation='linear', use_bias=False, 
                      kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, 
                      seed=42), kernel_regularizer=l2(l2_layer_reg))(layer_out)
                      
        network_out = Add()([residual, network_in])
        
        return network_out, skip_out
    
    return f


def DC_CNN_Model(length):
    
    input = Input(shape=(length,1))
    
    l1a, l1b = DC_CNN_Block(32,2,1,0.001)(input)    
    l2a, l2b = DC_CNN_Block(32,2,2,0.001)(l1a) 
    l3a, l3b = DC_CNN_Block(32,2,4,0.001)(l2a)
    l4a, l4b = DC_CNN_Block(32,2,8,0.001)(l3a)
    l5a, l5b = DC_CNN_Block(32,2,16,0.001)(l4a)
    l6a, l6b = DC_CNN_Block(32,2,32,0.001)(l5a)
    l6b = Dropout(0.8)(l6b) #dropout used to limit influence of earlier data
    l7a, l7b = DC_CNN_Block(32,2,64,0.001)(l6a)
    l7b = Dropout(0.8)(l7b) #dropout used to limit influence of earlier data

    l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
    
    l9 =   Activation('relu')(l8)
           
    l21 =  Conv1D(1,1, activation='linear', use_bias=False, 
           kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=42),
           kernel_regularizer=l2(0.001))(l9)

    model = Model(input=input, output=l21)
    
    adam = optimizers.Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, 
                           decay=0.0, amsgrad=False)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    
    return model


def evaluate_timeseries(timeseries, predict_size):
    # timeseries input is 1-D numpy array
    # forecast_size is the forecast horizon
    
    timeseries = timeseries[~pd.isna(timeseries)]

    length = len(timeseries)-1

    timeseries = np.atleast_2d(np.asarray(timeseries))
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T 

    model = DC_CNN_Model(length)
    print('\n\nModel with input size {}, output size {}'.
                                format(model.input_shape, model.output_shape))
    
    model.summary()

    X = timeseries[:-1].reshape(1,length,1)
    y = timeseries[1:].reshape(1,length,1)
    
    model.fit(X, y, epochs=3000)
    
    pred_array = np.zeros(predict_size).reshape(1,predict_size,1)
    X_test_initial = timeseries[1:].reshape(1,length,1)
    #pred_array = model.predict(X_test_initial) if predictions of training samples required
    
    #forecast is created by predicting next future value based on previous predictions
    pred_array[:,0,:] = model.predict(X_test_initial)[:,-1:,:]
    for i in range(predict_size-1):
        pred_array[:,i+1:,:] = model.predict(np.append(X_test_initial[:,i+1:,:], 
                               pred_array[:,:i+1,:]).reshape(1,length,1))[:,-1:,:]
    
    return pred_array.flatten()



    


