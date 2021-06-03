
import tensorflow as tf
from model.model import get_model_transfer, load_models
from utils.get_data import get_data

import argparse

parser = argparse.ArgumentParser(description='Traing the model')

parser.add_argument('--train_path', help='path to data with out labels', default='./registers/')
parser.add_argument('--train_path_label', help='path to data labels',default='./masks/')

parser.add_argument('--test_path', help='path to data with out labels',default='./registers/')
parser.add_argument('--test_path_label', help='path to data labels',default='./masks/')

parser.add_argument('--val_path', help='path to data with out labels',default='./registers/')
parser.add_argument('--val_path_label', help='path to data labels',default='./masks/')

parser.add_argument('--save_freq',help='epochs save', default=1)

parser.add_argument('--epochs',help='epochs training', default=1)

parser.add_argument('--fine_tune',help='epochs training', default=0)

args = parser.parse_args()

def Dice_loss(y_true,y_pred):
    nue = 2*tf.reduce_sum(y_true*y_pred)
    den = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2)
    return 1-nue/den

def get_callback(name,batch_size,epoch):
    checkpoint_path = f'./model/weights_{name}'+'/cp-{epoch:04d}.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=epoch*batch_size)
    return cp_callback

def main():
    train_dataset = args.train_path
    train_dataset_label = args.train_path_label
    test_dataset = args.test_path
    test_dataset_label = args.test_path_label
    val_dataset = args.val_path
    val_dataset_label = args.val_path_label
    epochs = args.epochs

    train_sagital = get_data(train_dataset,train_dataset_label, axis=0)
    test_sagital  =  get_data(test_dataset,test_dataset_label,axis=0)
    val_sagital  =  get_data(val_dataset,val_dataset_label,axis=0)

    train_coronal = get_data(train_dataset,train_dataset_label,axis=1)
    test_coronal  = get_data(test_dataset,test_dataset_label,axis=1)
    val_coronal  =  get_data(val_dataset,val_dataset_label,axis=1)

    train_axial = get_data(train_dataset,train_dataset_label,axis=2)
    test_axial  = get_data(test_dataset,test_dataset_label,axis=2)
    val_axial  =  get_data(val_dataset,val_dataset_label,axis=2)

    if args.fine_tune:
        print(f'{'='*10}fine_tune{'='*10}')
        model_sagital,model_coronal,model_axial = load_models()
    else:
        print(f'{'='*10}from_scratch{'='*10}')
        model_sagital = get_model_transfer(name='sagital')
        model_coronal = get_model_transfer(name='coronal')
        model_axial = get_model_transfer(name='axial')


    model_sagital.compile(loss=Dice_loss,metrics=tf.keras.metrics.BinaryAccuracy(),)
    model_coronal.compile(loss=Dice_loss,metrics=tf.keras.metrics.BinaryAccuracy())
    model_axial.compile(loss=Dice_loss,metrics=tf.keras.metrics.BinaryAccuracy())

    batch_sagital = int(train_sagital.reduce(0, lambda x, _: x + 1).numpy())
    batch_coronal = int(train_coronal.reduce(0, lambda x, _: x + 1).numpy())
    batch_axial = int(train_axial.reduce(0, lambda x, _: x + 1).numpy())

    cp_callback_sagital = get_callback('sagital',batch_sagital,args.save_freq)
    cp_callback_coronal = get_callback('coronal',batch_coronal,args.save_freq)
    cp_callback_axial = get_callback('axial', batch_axial,args.save_freq)


    history_sagital = model_sagital.fit(train_sagital,epochs=epochs,
                                   callbacks=[cp_callback_sagital,
                                   tf.keras.callbacks.TensorBoard(log_dir='./model/sagital_logs')],
                                    validation_data=val_sagital
                                    )

    history_coronal = model_coronal.fit(train_coronal,epochs=epochs,
                                   callbacks=[cp_callback_coronal,
                                   tf.keras.callbacks.TensorBoard(log_dir='./model/coronal_logs')],
                                    validation_data=val_coronal
                                    )
                                   
    history_axial   = model_axial.fit(train_axial,epochs=epochs,
                                   callbacks=[cp_callback_axial,
                                   tf.keras.callbacks.TensorBoard(log_dir='./model/axial_logs')],
                                    validation_data=val_axial
                                    )


if __name__ == "__main__":
    main()
    