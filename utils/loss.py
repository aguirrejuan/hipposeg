import tensorflow as tf
from scipy.ndimage import distance_transform_edt as eucl_distance
import numpy as np

def GDL_loss(y_true,y_pred):
    """
    Generalized Dice Loss
    y_true \in \mathbb{R}^{BxHxWxC}
    y_pred \in \mathbb{R}^{BxHxWxC}
    """
    B = tf.shape(y_true)[0]
    result = 0.0
    for b in range(B):
        if tf.reduce_sum(y_true[b]) == 0: 
            continue
        y,y_hat = y_true[b],y_pred[b]
        wG = 1/tf.einsum('ijk->',y)**2
        wB = 1/tf.einsum('ijk->',1-y)**2
        numerator = wG*tf.einsum('ijk,ijk->',y,y_hat) + wB* tf.einsum('ijk,ijk->',1-y,1-y_hat)
        denominator = wG*tf.einsum('ijk->',y+y_hat) + wB*tf.einsum('ijk->',2-y-y_hat)
        result += 1 -2*numerator/denominator

    return result/tf.cast(B,tf.float32)

def distmaps(y_true):
    k = tf.shape(y_true)[0]
    res = np.zeros_like(y_true,dtype=np.float32)
    for i in range(k):
        posmask = y_true[i,...] > 0.5
        negmask = ~ posmask
        res[i,...] = tf.cast(eucl_distance(negmask),dtype=tf.float32)* tf.cast(negmask,tf.float32) -tf.cast(eucl_distance(posmask),dtype=tf.float32)*tf.cast(posmask,tf.float32)
    return res

def boundary_loss(y_true,y_pred):
    """
    Boundary loss
    y_true \in \mathbb{R}^{BxHxW}
    y_pred \in \mathbb{R}^{BxHxW}
    """
    dist_map = tf.py_function(func=distmaps, inp=[y_true], Tout=tf.float32) # tensor [batch,h,w]
    value = tf.einsum('ijlk,ijlk->i',dist_map,y_pred) #- tf.einsum('ijk,ijk->i',dist_map,y_true)  # tensor[batch]
    return tf.reduce_mean(value)#/tf.reduce_sum(tf.abs(dist_map))

def Dice_metric(y_true,y_pred):
    B = tf.shape(y_true)[0]
    res = 0.0
    for b in range(B):
        if tf.reduce_sum(y_true[b]) == 0:
            continue 
        y, y_hat = y_true[b] , y_pred[b]
        intersection = tf.einsum('ijk,ijk->',y,y_hat) #haddamar -> sum j and k for each image
        union = tf.einsum('ijk->',y**2) + tf.einsum('ijk->',y_hat**2) #sum  j and k
        res +=  2*intersection/union # dice per image 
    return res/tf.cast(B,tf.float32)

def Dice_loss(y_true,y_pred):
    nue = 2*tf.reduce_sum(y_true*y_pred)
    den = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2)
    return 1-nue/den
