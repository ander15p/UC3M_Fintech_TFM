
"""
Script con los modelos de redes neuronales, función de coste y scaler
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from  tensorflow.keras.metrics import mean_squared_error
import pandas as pd


tf.keras.backend.set_floatx('float32') 
real_type = tf.float32

class basic_scaler:
    def __init__(self,x,y,z,z_true = True):#z = dydx
        self.x_mean =x.numpy().mean(axis=0)
        self.y_mean = y.numpy().mean(axis=0)
        self.x_std = x.numpy().std(axis=0)
        self.y_std = y.numpy().std(axis=0)
        if z_true == True:
            z_scaled = self.adapt(x,y,z)[2]
            self.CustomLoss = CustomLoss(z_scaled)
        
    def adapt(self,x,y,z):
        x_scaled = (x-self.x_mean)/self.x_std
        y_scaled = (y-self.y_mean)/self.y_std
        z_scaled = z*self.x_std/self.y_std

        return x_scaled,y_scaled,z_scaled

    def unscale(self,x_scaled,y_scaled,z_scaled):
        x = x_scaled * self.x_std + self.x_mean
        y = y_scaled * self.y_std + self.y_mean
        z = z_scaled * self.y_std / self.x_std 
        return x,y,z
    def unscale_y(self,y_scaled):
        y = y_scaled * self.y_std + self.y_mean 
        return y
    def unscale_dydx(self,dydx_scaled):
        z = dydx_scaled * self.y_std / self.x_std 
        return z
    def adapt_x(self,x):
        x_scaled = (x-self.x_mean)/self.x_std
        return x_scaled
    
    
def predict_values(model,scaler,x,diffs = False):
    x_scaled = scaler.adapt_x(x)
    y_scaled,z_scaled = model.predict(x_scaled)
    
    if diffs == False:
        return scaler.unscale(0,y_scaled,0)[1]
    if diffs == True:
        return scaler.unscale(0,y_scaled,z_scaled)[1:]

def predict_values_vanilla(model,scaler,x):
    x_scaled = scaler.adapt_x(x)
    y = scaler.unscale_y(model.predict(x_scaled))
    return y


class PrintProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 10 == 0: 
          print("Epoch {:03d}".format(epoch))

class CustomLoss:
    def __init__(self,dydx):
        self.lambda_j = 1.0 / np.sqrt((dydx.numpy() ** 2).mean(axis=0)).reshape(1, -1)        
        
    def function_derivs(self,y_true,y_pred):
        return tf. losses.mean_squared_error(y_true * self.lambda_j, y_pred * self.lambda_j)

class BackpropDense(tf.keras.layers.Layer):
    def __init__(self, reference_layer, **kwargs):
      super(BackpropDense, self).__init__(**kwargs)
      self.units = None
      self.ref_layer = reference_layer # weights of ref layer 'collected' by tensorflow
    
    def call(self, gradient, z):
        if z is not None:

            gradient = tf.matmul(gradient, tf.transpose(self.weights[0])) * tf.math.sigmoid(z)
        else:
            gradient = tf.matmul(gradient, tf.transpose(self.weights[0]))
        return gradient
    
    def get_config(self):
        config = super(BackpropDense, self).get_config()
        config.update({"reference_layer": self.ref_layer})
        return config

def get_vanilla_net(input_dim):
    input_1 = layers.Input(shape=(input_dim,))
    layer_1 = layers.Dense(20, kernel_initializer='glorot_normal', activation = 'softplus', name='FWD_L1')(input_1)
    layer_2 = layers.Dense(20, kernel_initializer='glorot_normal', activation = 'softplus', name='FWD_L2')(layer_1)
    layer_3 = layers.Dense(20, kernel_initializer='glorot_normal', activation = 'softplus', name='FWD_L3')(layer_2)
    layer_4 = layers.Dense(20, kernel_initializer='glorot_normal', activation = 'softplus',name='FWD_L4')(layer_3)
    y_pred = layers.Dense(1, kernel_initializer='glorot_normal',name='y_pred')(layer_4)
    model = tf.keras.models.Model(inputs=input_1, outputs=y_pred, name='Vanilla_Net')
    return model

def get_twin_net(input_dim):


    layer_1 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L1')
    layer_2 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L2')
    layer_3 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L3')
    layer_4 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L4')
    layer_5 = layers.Dense(1, kernel_initializer='glorot_normal', activation = 'linear', name='y_pred')


    input_1 = layers.Input(shape=(input_dim,))
    x1 = layer_1(input_1)
    x2 = layer_2(layers.Activation('softplus', name="Act_1")(x1))
    x3 = layer_3(layers.Activation('softplus', name="Act_2")(x2))
    x4 = layer_4(layers.Activation('softplus', name="Act_3")(x3))
    y_pred = layer_5(layers.Activation('softplus', name="Act_4")(x4))


    grad = BackpropDense(layer_5,name='Bck_L1')(tf.ones_like(y_pred), x4)
    grad = BackpropDense(layer_4,name='Bck_L2')(grad, x3)
    grad = BackpropDense(layer_3,name='Bck_L3')(grad, x2)
    grad = BackpropDense(layer_2,name='Bck_L4')(grad, x1)
    dydx_pred = BackpropDense(layer_1,name='dydx_pred')(grad, None)

    model = tf.keras.models.Model(inputs=input_1, outputs=[y_pred, dydx_pred], name='Twin_Net')

    return model