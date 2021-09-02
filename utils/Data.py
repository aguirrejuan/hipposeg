"""
Data.py 

Module for preprocessing the input data for the model e2dhipposeg.

"""
 
import tensorflow as tf
import tensorflow_addons as tfa
import nibabel as nib 
from glob import glob
import numpy as np
import random

_size_crop = 160

random.seed(42)
tf.random.set_seed(42)


def get_slides(X,axis,size_crop=_size_crop):
    """ Return generator with slides to perform inference
    Parameters
        X: Tensor 
        axis: Axis where we can to pass the slides
        size_crop: size of crop
    Returns
        generator 
    """
    def generator():
        for i in range(tf.shape(X)[axis]):
            index = [i if j < 0 or j >= tf.shape(X)[axis] else j for j in range(i-1,i+2)]
            x_1 = X[(slice(None),)*axis+(index[0],)]
            x_2 = X[(slice(None),)*axis+(index[1],)]
            x_3 = X[(slice(None),)*axis+(index[2],)]
            x = [tf.expand_dims(x_i,axis=2) for x_i in [x_1,x_2,x_3]]
            yield crop(tf.concat(x,axis=2),size_crop=size_crop)
    return generator
        
def load_mri(vol_path,size_crop=_size_crop):
    """ Returns slides for each axis to perform inference
    Parameters 
        vol_path: path to MRI 
        size_crop: size of crop 
    """
    X = tf.constant(nib.load(vol_path).get_fdata())
    X = minmax_normalization(X)
    shape = X.shape
    return get_slides(X,0,size_crop),get_slides(X,1,size_crop),get_slides(X,2,size_crop),shape


def minmax_norm(X):
    """ Min Max normalization
    Parameter
        X: tf.tensor 
    Rerurns
        tf.tensor
    """
    return (X - tf.reduce_min(X))/(tf.reduce_max(X)-tf.reduce_min(X)) 

def crop(X,size_crop=_size_crop):
    """ Crop the input tensor with center in the middle 
    Parameters
        X: Tensor [H,W,C]
        size_crop: size of cropped tensor 
    Returns 
        X: Tensor [H//size_crop,W//_size_crop,C]
    """
    b = size_crop//2
    shape = tf.shape(X)
    cx= shape[0]//2
    cy= shape[1]//2
    return X[cx-b:cx+b,cy-b:cy+b,...]


def crop_random(X,Y,random_crop=False,size_crop=_size_crop):
    """ Crop the input tensors with randomly center within the set where the 
        output dimentions fit. 
    Parameters
        X: Tensor MRI [H,W,C]
        Y: Tensor mask [H,W,C]
        random_crop: bool 
        size_crop: size of cropped tensor 
    Returns
        X and Y cropped tensors if it's the case
    """
    b = size_crop//2
    shape = tf.shape(X)
    if random_crop: 
        cx = tf.random.uniform(shape=(1,),minval=b,maxval=(shape[0]-b),dtype=tf.int32)[0]
        cy = tf.random.uniform(shape=(1,),minval=b,maxval=(shape[1]-b),dtype=tf.int32)[0]
        return X[cx-b:cx+b,cy-b:cy+b,...], Y[cx-b:cx+b,cy-b:cy+b,...]
    else: 
        return crop(X,size_crop=size_crop),crop(Y,size_crop=size_crop)



def generator_2d(path_imgs,path_masks, axis=0):
    """ Return a slides of MRIs
    Parameters
        path_img: Path to MRIs 
        path_mask: Path to masks of MRI 
        axis: Axis where we can to extract the slides 
    Returns
        generator that returns x[H,W,C=3] and y[X,W,C=1] slides
    """
    imgs =  sorted(glob(path_imgs))
    masks = sorted(glob(path_masks))

    def generator():

        for img,mask in zip(imgs,masks):

            X = nib.load(img).get_fdata()
            Y = nib.load(mask).get_fdata()
            X,Y = minmax_norm(tf.constant(X)),tf.constant(Y)

            range_ = range(tf.shape(X)[axis])

            for i in random.sample(range_,len(range_)):
                index = [i if j < 0 or j >= tf.shape(X)[axis] else j for j in range(i-1,i+2)]
                x_1   =   X[(slice(None),)*axis+(index[0],)]
                x_2,y =   X[(slice(None),)*axis+(index[1],)], Y[(slice(None),)*axis+(index[1],)]
                x_3   =   X[(slice(None),)*axis+(index[2],)]
                x     =   tf.stack([x_1,x_2,x_3],axis=-1)
                yield x, y[...,tf.newaxis]
    return generator


def intensity_modification(x):
    """ Intensity modification 
    Parameters
      x: Tensor 
    Returns
      x: Tensor 
    """
    x = x + tf.random.uniform(shape=[], minval=-0.05, maxval=0.05, dtype=tf.dtypes.float32)
    return x

def gaussian_noise(x):
    """ Add gaussian noise
    Parameters
      x: Tensor 
    Returns 
      x: Tensor
    """
    x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.0002, dtype=tf.dtypes.float32)
    return x

def rotation_and_scale(x,y):
    """ Ramdonly rotation and scale 
    Parameters
      x: MRI tensor
      y: Mask MRI tensor 
    Returns 
      x: MRI tensor rotated and scaled
      y: Mask MRR tensor rotated and scaled
    """
    scale = tf.random.uniform(shape=[], minval=0.85, maxval=1.2, dtype=tf.dtypes.float32)
    angle = tf.random.uniform(shape=[], minval=-10*np.pi/180, maxval=10*np.pi/180)
    
    shape = tf.cast(tf.shape(x),tf.float32)
    scale = tf.cast([scale*shape[0],scale*shape[1]],tf.float32)
    
    x = tf.image.resize(x,[scale[0],scale[1]])
    y = tf.image.resize(y,[scale[0],scale[1]])
    x = tfa.image.transform_ops.rotate(x, angle)
    y = tfa.image.transform_ops.rotate(y, angle) > 0.5

    return x,tf.cast(y ,tf.float32)

def get_hippo(pixels):
    """ Return filter 
    """
    def filter(x,y):
        """ Filter image if it has more than n pixels equal to one in the mask
        """
        if tf.reduce_sum(y) > pixels:
            return True
        else:
            return False 
    return filter


def embedded_func(x,y,random_crop=False,size_crop=_size_crop):
    """ Embedded data augmentation funcitons 
    """
    x,y = rotation_and_scale(x,y)
    x,y = crop_random(x,y,random_crop=random_crop,size_crop=size_crop)
    x = intensity_modification(x)
    x = gaussian_noise(x)
    return x,y 


def get_data(path_imgs,path_masks,axis=0,
             batch=50,buffer_size=300,
             size_crop=160,random_crop=False,
             augmentation=False,repeat=1,
             pixels=0,cache=True,
             ): 
    """ Returns dataset already preprocessed
    """
    data = tf.data.Dataset.from_generator(generator_2d(path_imgs,path_masks,axis=axis),
                                    output_signature = (tf.TensorSpec((None,None,None), tf.float32), 
                                                        tf.TensorSpec((None,None,None), tf.float32)))
    if augmentation:
      data = data.repeat(repeat)
      
      data = data.map(lambda x,y : (embedded_func(x,y,random_crop=random_crop,size_crop=size_crop)),
                        num_parallel_calls=tf.data.AUTOTUNE)
      
    else: 
      data = data.map(lambda X,Y: crop_random(X,Y,random_crop=random_crop,size_crop=size_crop),
                    num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
      data = data.cache()

    if pixels != 0:
      data = data.filter(get_hippo(pixels=pixels))

    data = data.shuffle(buffer_size=buffer_size, seed=42)
    data = data.batch(batch)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data 
