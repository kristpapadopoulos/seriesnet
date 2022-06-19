from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Add, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import TruncatedNormal


class DcCnnBlock(layers.Layer):
    """
    Tensorflow implementation of Dilated Causal Convolutional Neural Network Block 
    for SeriesNet model for predicting time series.
    
    Args:
      num_filter: int - the number of convolution filters in DcCnnBlock
      dilation: int - the lookback window for each DcCnnBlock
      filter_length: int - the length of convolution filters in DcCnnBlock
      l2_layer_reg: float - l2 layer weight regularization
    """
    def __init__(self, 
                 num_filter, 
                 filter_length, 
                 dilation, 
                 l2_layer_reg, 
                 name="dc_cnn_block", 
                 **kwargs):
        
        super(DcCnnBlock, self).__init__(name=name, **kwargs)
        
        self.layer_out = Conv1D(filters=num_filter, 
                                kernel_size=filter_length, 
                                dilation_rate=dilation,
                                activation='linear', 
                                padding='causal', 
                                use_bias=False,
                                kernel_initializer=TruncatedNormal(mean=0.0, 
                                                                   stddev=0.05,
                                                                   seed=42), 
                                kernel_regularizer=l2(l2_layer_reg))
        
        self.act = Activation('selu')
        
        self.skip_out = Conv1D(1,
                               1, 
                               activation='linear', 
                               use_bias=False, 
                               kernel_initializer=TruncatedNormal(mean=0.0, 
                                                                 stddev=0.05,
                                                                  seed=42), 
                               kernel_regularizer=l2(l2_layer_reg))
        
        self.network_in = Conv1D(1,
                                 1, 
                                 activation='linear', 
                                 use_bias=False,
                                 kernel_initializer=TruncatedNormal(mean=0.0, 
                                                                    stddev=0.05,
                                                                    seed=42), 
                                 kernel_regularizer=l2(l2_layer_reg))

    def call(self, inputs):
        residual = inputs
        layer_out = self.layer_out(inputs)
        layer_out = self.act(layer_out)
        skip_out = self.skip_out(layer_out)
        network_in = self.network_in(layer_out)
        network_out = Add()([residual, network_in])
        return network_out, skip_out

      
class SeriesNet(keras.Model):
    """
    Tensorflow implementation of Dilated Causal Convolutional Neural Network for Time 
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
    
    V1 Date: September 12, 2018
        - updated Keras merge function to Add for Keras 2.2.2
         tensorflow==1.10.1
         Keras==2.2.2
         numpy==1.14.5
         
    V1.1 Date: June 19, 2022
        - updated to tf.keras layer/model format
        tensorflow==2.6.5
        numpy==1.19.5
    
    Args:
        num_filter: int - the number of convolution filters in DcCnnBlock
        filter_length: int - the length of convolution filters in DcCnnBlock
        l2_layer_reg: float - l2 layer weight regularization
        dropout: float - dropout fraction
    """
    def __init__(
        self,
        num_filter, 
        filter_length, 
        l2_layer_reg,
        dropout,
        name="dc_cnn_model",
        **kwargs):
        
        super(SeriesNet, self).__init__(name=name, **kwargs)
        
        self.block1 = DcCnnBlock(num_filter, filter_length, 1, l2_layer_reg)
        self.block2 = DcCnnBlock(num_filter, filter_length, 2, l2_layer_reg)
        self.block3 = DcCnnBlock(num_filter, filter_length, 4, l2_layer_reg)
        self.block4 = DcCnnBlock(num_filter, filter_length, 8, l2_layer_reg)
        self.block5 = DcCnnBlock(num_filter, filter_length, 16, l2_layer_reg)
        self.block6 = DcCnnBlock(num_filter, filter_length, 32, l2_layer_reg)
        self.block7 = DcCnnBlock(num_filter, filter_length, 64, l2_layer_reg)
        self.dropout = Dropout(dropout)
        self.act = Activation('relu')
        self.out = Conv1D(1,
                          1, 
                          activation='linear', 
                          use_bias=False, 
                          kernel_initializer=TruncatedNormal(mean=0.0, 
                                                             stddev=0.05, 
                                                             seed=42),
                          kernel_regularizer=l2(l2_layer_reg))

    def call(self, inputs):
        l1a, l1b = self.block1(inputs)    
        l2a, l2b = self.block2(l1a) 
        l3a, l3b = self.block3(l2a)
        l4a, l4b = self.block4(l3a)
        l5a, l5b = self.block5(l4a)
        l6a, l6b = self.block6(l5a)
        l6b = self.dropout(l6b) #dropout used to limit influence of earlier data
        l7a, l7b = self.block7(l6a)
        l7b = self.dropout(l7b) #dropout used to limit influence of earlier data
        l8 = Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
        l9 = self.act(l8)
        l21 = self.out(l9)
        return l21
      
