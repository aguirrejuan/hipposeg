import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import nibabel as nib
from glob import glob 
import os 
from .config import cfg

def minmax_normalization(x):
    return (x-tf.reduce_min(x))/(tf.reduce_max(x)-tf.reduce_min(x))

def crop(X):
    cx= tf.shape(X)[0]//2
    cy= tf.shape(X)[1]//2
    b = cfg.CROP//2
    return X[cx-b:cx+b,cy-b:cy+b,...]

def get_slides(X,axis):
    def generator():
        for i in range(tf.shape(X)[axis]):
            index = [i if j < 0 or j >= tf.shape(X)[axis] else j for j in range(i-1,i+2)]
            x_1 = X[(slice(None),)*axis+(index[0],)]
            x_2 = X[(slice(None),)*axis+(index[1],)]
            x_3 = X[(slice(None),)*axis+(index[2],)]
            x = [tf.expand_dims(x_i,axis=2) for x_i in [x_1,x_2,x_3]]
            yield crop(tf.concat(x,axis=2))
    return generator
        
def load_mri(vol_path):
    X = tf.constant(nib.load(vol_path).get_fdata(), "float64")
    X = minmax_normalization(X)
    shape = X.shape
    return get_slides(X,0),get_slides(X,1),get_slides(X,2),shape

def generator_load(path_mri,path_label):
    mris = sorted(glob(os.path.join(path_mri,'*.nii')))
    labels = sorted(glob(os.path.join(path_label,'*.nii')))
    def generator ():
        for X,Y in zip(mris,labels):
            yield tf.constant(nib.load(X).get_fdata(), "float64"), tf.constant(nib.load(Y).get_fdata(), "float64")
    return generator

def generator_2D(dataset,axis=0):
    def generator():
        for X,Y in dataset:
            for i in range(tf.shape(X)[axis]):
                index = [i if j < 0 or j >= tf.shape(X)[axis] else j for j in range(i-1,i+2)]
                x_1,y_1 = X[(slice(None),)*axis+(index[0],)], Y[(slice(None),)*axis+(index[0],)]
                x_2,y_2 = X[(slice(None),)*axis+(index[1],)], Y[(slice(None),)*axis+(index[1],)]
                x_3,y_3 = X[(slice(None),)*axis+(index[2],)], Y[(slice(None),)*axis+(index[2],)]
                x = [tf.expand_dims(x_i,axis=2) for x_i in [x_1,x_2,x_3]]
                yield tf.concat(x,axis=2),y_2
    return generator 

def get_data_2d(path_mri,path_label,axis=0):
    data = tf.data.Dataset.from_generator(generator_load(path_mri,path_label),
                                    output_signature = (tf.TensorSpec((None, None, None), tf.float64), 
                                                        tf.TensorSpec((None, None, None), tf.float64)))
    data = data.map(lambda x,y:(minmax_normalization(x),y))
    data2D = tf.data.Dataset.from_generator(generator_2D(data,axis=axis),
                                    output_signature = (tf.TensorSpec((None, None,None), tf.float64), 
                                                        tf.TensorSpec((None,None), tf.float64)))
    data2D = data2D.map(lambda x,y:(crop(x),crop(y)))
    return data2D

def get_data(dataset,dataset_label,
             axis,batch=50,buffer_size=100,prefetch=10):
    data = get_data_2d(dataset,dataset_label,axis=axis)
    data = data.shuffle(buffer_size=buffer_size)
    data = data.batch(batch)
    data = data.prefetch(prefetch)
    return data

if __name__ == "__main__":
    train_dataset = '../ADNI_HARP/images_100/'
    train_dataset_label = '../ADNI_HARP/Labels_100/'
    train_data = get_data(train_dataset,train_dataset_label,axis=0,batch=1)
    for data in train_data.take(1):
        plt.subplot(1,2,1)
        plt.imshow(data[1][20,...])
        plt.subplot(1,2,2)
        plt.imshow(data[0][20,...])
