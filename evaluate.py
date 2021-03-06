
from utils.metrics import jaccard,dice,precision,recall
from inference import predict
import numpy as np
import tensorflow as tf
from glob import glob 
import os
import argparse
import nibabel as nib

def metrics(y,y_pred):
    return [jaccard(y,y_pred),dice(y,y_pred),precision(y,y_pred),recall(y,y_pred)]

def scores(data,data_label):
    scores = []
    for x,y in zip(data,data_label):
        y = nib.load(y).get_fdata()
        y_pred =  tf.cast(tf.squeeze(predict(x)) >0.5,dtype=tf.float64)
        y = tf.cast(tf.squeeze(y) >0.5,dtype=tf.float64)
        scores.append(metrics(y,y_pred))
    return np.array(scores).mean(axis=0),np.array(scores).std(axis=0)

def print_metrics(name,scores):
    print(name.upper())
    for i,mt in enumerate(['Jaccar','Dice','Precision','Recall']):
        print(f'{mt:<10}:{scores[0][i]:.3f} +/- {scores[1][i]:.3f}')
    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--path_data',help='path to data')
    parser.add_argument('--path_data_label',help='path to data label')
    args = parser.parse_args()

    data = sorted(glob(os.path.join(args.path_data,'*.nii')))
    data_label = sorted(glob(os.path.join(args.path_data_label,'*.nii')))
    scr = scores(data,data_label)
    print_metrics('results',scr)
