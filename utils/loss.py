import tensorflow as tf
from scipy.ndimage import distance_transform_edt as eucl_distance


def distMaps(seg):
    posmask = seg > 0.5
    negmask = ~ posmask
    return eucl_distance(negmask)* negmask - (eucl_distance(posmask) - 1) * posmask

def boundary_loss(y_true,y_pred):
    dist_map = tf.py_function(func=distMaps, inp=[y_true], Tout=tf.float32)
    value = (dist_map*x_pred - dist_map*x)
    return tf.reduce_sum(value)

def Dice_loss(y_true,y_pred):
    nue = 2*tf.reduce_sum(y_true*y_pred)
    den = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2)
    return 1-nue/den
