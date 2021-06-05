import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

def plot_predict(x,y,y_pred):
    """
    (None,None,3),(None,None)(bool),(None,None)(bool)
    """
    plt.figure(figsize=(6,6))
    x,y,y_pred = x[...,np.newaxis],y[...,np.newaxis],y_pred[...,np.newaxis]
    mask  = x * (1 - tf.cast(tf.math.logical_or(y,y_pred),dtype=tf.float64))
    Rojo  = mask + tf.cast(tf.math.logical_and(~y,y_pred),dtype=tf.float64)  #false positive
    Verde = mask + tf.cast(tf.math.logical_and(y,y_pred),dtype=tf.float64)  #true positive
    Azul  = mask + tf.cast(tf.math.logical_and(y,~y_pred),dtype=tf.float64)  #false negative
    image =  np.concatenate((Rojo,Verde,Azul),axis=2)
    plt.imshow(image)
    plt.show()