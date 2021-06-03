from tqdm import tqdm
from utils.config import cfg
from utils.get_data import load_mri
from model.model import load_models
import nibabel as nib
import numpy as np
import tensorflow as tf

from datetime import datetime

import argparse 

parser = argparse.ArgumentParser(description='Traing the model')
parser.add_argument('--path_vol',help='path to volume')

args = parser.parse_args()


def predict_axis(model,generator,shape):
    vol = np.zeros((shape,cfg.CROP,cfg.CROP))
    for i in tqdm(range(shape)):
        sag = next(generator)
        vol[i,...]= tf.squeeze(model.predict(sag[np.newaxis,...]))
    return vol


def predict(vol_path):
    sagital,coronal,axial,shape = load_mri(vol_path)
    model_sagital,model_coronal,model_axial = load_models()
    vol_sagital = predict_axis(model_sagital,sagital(),shape[0])
    vol_coronal = predict_axis(model_coronal,coronal(),shape[1])
    vol_axial = predict_axis(model_axial,axial(),shape[2])
    
    CROP =  cfg.CROP
    pad_0 = (shape[0]-CROP)//2 - 1
    pad_1 = (shape[1]-CROP)//2 - 1
    pad_2 = (shape[2]-CROP)//2 - 1
    vol_sagital = tf.pad(vol_sagital,[[0,0],[pad_1,shape[1]-(pad_1+CROP)],[pad_2,shape[2]-(pad_2+CROP)]])
    vol_coronal = tf.pad(tf.transpose(vol_coronal,[1,0,2]),[[pad_0,shape[0]-(pad_0+CROP)],[0,0],[pad_2,shape[2]-(pad_2+CROP)]])
    vol_axial = tf.pad(tf.transpose(vol_axial,[1,2,0]),[[pad_0,shape[0]-(pad_0+CROP)],[pad_1,shape[1]-(pad_1+CROP)],[0,0],])

    output = vol_sagital+ vol_coronal + vol_axial
    return output/3

if __name__ == "__main__":
    vol_path = args.path_vol
    vol = predict(vol_path).numpy()
    vol = nib.Nifti1Image(vol, np.eye(4))
    nib.save(vol,f'./result{str(datetime.now())[:-7]}.nii')
