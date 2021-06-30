import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

def plot_predict(x,y,y_pred):
    """
    (None,None,3),(None,None,1)(bool),(None,None,1)(bool)
    """
    plt.figure(figsize=(6,6))
    mask  = x * (1 - tf.cast(tf.math.logical_or(y,y_pred),dtype=tf.float32))
    Rojo  = mask + tf.cast(tf.math.logical_and(~y,y_pred),dtype=tf.float32)  #false positive
    Verde = mask + tf.cast(tf.math.logical_and(y,y_pred),dtype=tf.float32)  #true positive
    Azul  = mask + tf.cast(tf.math.logical_and(y,~y_pred),dtype=tf.float32)  #false negative
    image =  np.concatenate((Rojo,Verde,Azul),axis=2)
    plt.imshow(image)
    plt.show()
