import tensorflow as tf
from scipy.ndimage import distance_transform_edt as eucl_distance


def distMaps(seg):
    k = tf.shape(seg)[0]
    res = tf.zeros_like(seg,dtype=tf.float32)
    for i in range(k):
        posmask = seg[i,...] > 0.5
        negmask = ~ posmask
        res[i,...] = tf.cast(eucl_distance(negmask),dtype=tf.float32)*tf.cast(negmask,tf.float32) - (tf.cast(eucl_distance(posmask),tf.float32) - 1) * tf.cast(posmask,tf.float32)
    return res

def boundary_loss(y_true,y_pred):
    dist_map = tf.py_function(func=distMaps, inp=[y_true], Tout=tf.float32)
    value = (dist_map*y_pred - dist_map*y_true)
    return tf.reduce_sum(value)

def Dice_loss(y_true,y_pred):
    nue = 2*tf.reduce_sum(y_true*y_pred)
    den = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2)
    return 1-nue/den
