
import tensorflow as tf
import logging
from model.model import get_model_transfer, load_models,load_model
from utils.get_data import get_data
from evaluate import print_metrics,scores
from glob import glob 
import os 
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='Traing the model')

parser.add_argument('--train_path', help='path to data with out labels', default='./registers/')
parser.add_argument('--train_path_label', help='path to data labels',default='./masks/')

parser.add_argument('--test_path', help='path to data with out labels',default=None)
parser.add_argument('--test_path_label', help='path to data labels',default= None)

parser.add_argument('--val_path', help='path to data with out labels',default=None)
parser.add_argument('--val_path_label', help='path to data labels',default=None)

parser.add_argument('--save_freq',help='epochs save', default=1 ,type=int)

parser.add_argument('--epochs',help='epochs training', default=1, type=int)

parser.add_argument('--batch_size',help='epochs training', default=50, type=int)

parser.add_argument('--repeat',help='epochs training', default=1, type=int)

parser.add_argument('--fine_tune',help='epochs training',action='store_true')

parser.add_argument('--evaluate',help='evaluate Model', action='store_true')

parser.add_argument('--models',help='evaluate Model', default='012')


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
    batch_size = args.batch_size
    repeat = args.repeat


    tf.random.set_seed(42)

    try:
        tpu= tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except:
        tpu = None
    if tpu: 
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else: 
        strategy = tf.distribute.get_strategy()

    models = ['sagital','coronal','axial']
    for i in range(3):
        if str(i) not in args.models:
            continue

        train =  get_data(train_dataset,train_dataset_label, axis=i,batch=batch_size,repeat=repeat)
        #test  =  get_data(test_dataset,test_dataset_label,axis=i)
        val  =   get_data(val_dataset,val_dataset_label,axis=i,repeat=repeat) if val_dataset != None else None 

        logging.info('Loading Model')
        with strategy.scope():
            if args.fine_tune:
                logging.info('Loading Model for Fine Tune')
                model = load_model(name=models[i],dir_weights=f'./model/weights_{models[i]}/')
            else:
                logging.info('Loading Model From scratch')
                model = get_model_transfer(name=models[i])
            model.compile(loss=Dice_loss,metrics=tf.keras.metrics.BinaryAccuracy(),)

        logging.info('Get cardinality of dataset(in slides for each axis)')
        list_data = glob(os.path.join(args.train_path,'*.nii'))
        lenght_data = len(list_data)
        shape = nib.load(list_data[0]).get_fdata().shape
        batch_epoch = repeat*shape[i]*lenght_data//batch_size#int(train_sagital.reduce(0, lambda x, _: x + 1).numpy())

        cp_callback = get_callback(models[i],batch_epoch,args.save_freq)


        logging.info('Trainig')
        history = model.fit(train,epochs=epochs,
                                   callbacks=[cp_callback,
                                   tf.keras.callbacks.TensorBoard(log_dir=f'./model/{models[i]}_logs')],
                                    validation_data=val
                                    )

    if args.evaluate and test_dataset != None and '012' in args.models:                                
        logging.info('evaluating...')
        data = sorted(glob(os.path.join(test_dataset,'*.nii')))
        data_label = sorted(glob(os.path.join(test_dataset_label,'*.nii')))
        scr = scores(data,data_label)
        print_metrics('results',scr)

if __name__ == "__main__":
    main()
    
