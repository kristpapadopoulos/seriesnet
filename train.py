import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from model.seriesnet import SeriesNet


def prepare_timeseries(timeseries):
    """
    Prepare timeseries for SeriesNet training and predictions.
    
    Args:
      timeseries: numpy array - 1-D array of values
    """
    timeseries = timeseries[~pd.isna(timeseries)]
    length = len(timeseries)-1

    timeseries = np.atleast_2d(np.asarray(timeseries))
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T
        
    X_train = timeseries[:-1].reshape(1,length,1)
    y_train = timeseries[1:].reshape(1,length,1)
        
    return X_train, y_train
  

def train(model, X_train, y_train, learning_rate = 0.00075, epochs = 3000):
    """
    Train SeriesNet model.
    
    Args:
      model: tensorflow model - SeriesNet model
      X_train: numpy array - input timeseries for training
      y_train: numpy array - target timeseries for training
      learning_rate: float - optimizer learning rate
      epochs: int - number of training loops
    """
    adam = Adam(learning_rate=learning_rate, 
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=None, 
                decay=0.0, 
                amsgrad=False)

    model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    model.fit(X_train, y_train, epochs = epochs)
    return model
  

def predict(model, y_train, predict_size):
    """
    Create predictions from trained SeriesNet model.
    
    Args:
      model: tensorflow model - trained SeriesNet model
      y_train: numpy array - time series array to create predictions
      predict_size: int - forecast window for predictions
    """
    pred_array = np.zeros(predict_size).reshape(1,predict_size,1)
    
    #forecast is created by predicting next future value based on previous predictions
    pred_array[:,0,:] = model.predict(y_train)[:,-1:,:]
    for i in range(predict_size-1):
        pred_array[:,i+1:,:] = model.predict(np.append(y_train[:,i+1:,:], 
                               pred_array[:,:i+1,:]).reshape(1,y_train.shape[1],1))[:,-1:,:]
    
    return pred_array.flatten()
  
  
if __name__ == "__main__":
    
    # example timeseries
    ts = np.log(range(1,108)) + np.random.normal(0, 0.5,107)
    
    # prepare timeseries
    X_train, y_train = prepare_timeseries(ts)
    
    # create SeriesNet model
    model = SeriesNet(32,2,0.001,0.8)
    
    # train model
    trained_model = train(model, X_train, y_train)
    
    # create forecast over 30 timesteps
    forecast_window = 30
    pred = predict(trained_model, y_train, forecast_window)
    
    # example plots for original timeseries and predictions
    plt.plot(list(range(len(ts))), ts)
    plt.plot(list(range(len(ts), len(ts)+forecast_window)), pred)
 
