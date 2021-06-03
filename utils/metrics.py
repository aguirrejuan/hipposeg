import tensorflow as tf 

def jaccard(y_true,y_pred):
    inter = tf.reduce_sum(y_true*y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return inter/union

def dice(y_true,y_pred):
    nue = 2*tf.reduce_sum(y_true*y_pred)
    den = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return nue/den

def precision(y_true,y_pred):
    inter = tf.reduce_sum(y_true*y_pred)
    estimate =tf.reduce_sum(y_pred)
    return inter/estimate

def recall(y_true,y_pred):
    inter = tf.reduce_sum(y_true*y_pred)
    mask =tf.reduce_sum(y_true)
    return inter/mask