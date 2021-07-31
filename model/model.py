from tensorflow.keras import Model
import tensorflow as tf
from glob import glob



def kernel_initializer(seed=42):
  return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)


from tensorflow.keras.layers import (
        Conv2D,
        Conv2DTranspose,
        BatchNormalization,
        Input,
        Add,
        ReLU,
        MaxPool2D,
        concatenate,
        Softmax)

def transfer(model,vgg_path='./vgg11'):
    vgg11 = tf.keras.models.load_model(vgg_path)
    layersVgg = vgg11.layers 
    for layerVgg in layersVgg:
        for layerModel in model.layers[1].layers:
            try:
                if layerVgg.get_weights()[0].shape == layerModel.get_weights()[0].shape:
                    layerModel.set_weights([layerVgg.get_weights()[0]])
                    layerModel.trainable = False
                    break
            except:
                pass

def convBlock(filters):
    def block(x):
        x_i = x
        x = Conv2D(filters=filters,kernel_size=3,padding='same',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer(seed=32))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=filters,kernel_size=3,padding='same',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer(seed=45))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x_i = Conv2D(filters=filters,kernel_size=1,padding='same',
                                                  use_bias=False,
                                                  kernel_initializer=kernel_initializer(seed=76))(x_i)
        x = Add()([x,x_i])
        return x 
    return block

def go_down(name='encoder'):
    x = input_ = Input([None,None,3])
    x = level1 = convBlock(filters=64)(x)
    x = MaxPool2D()(x)

    x = level2 = convBlock(filters=128)(x)
    x = MaxPool2D()(x)

    x = level3 = convBlock(filters=256)(x)
    x = MaxPool2D()(x)

    x = level4 = convBlock(filters=512)(x)
    x = MaxPool2D()(x)

    x = convBlock(filters=512)(x)
    return Model(input_,(x,level4,level3,level2,level1),name=name)
    
def go_up(name='decoder'):
    shape = [512,512,256,128,64]
    X = input_ = [Input(shape=[None,None,i]) for i in shape]
    x = Conv2DTranspose(filters=512,kernel_size=2,strides=2,
                                              use_bias=False,
                                              kernel_initializer=kernel_initializer(seed=87))(X[0])
    x = concatenate([x,X[1]])
    x = convBlock(filters=256)(x)

    x = Conv2DTranspose(filters=256,kernel_size=2,strides=2,
                                                  use_bias=False,
                                                  kernel_initializer=kernel_initializer(seed=98))(x)
    x = concatenate([x,X[2]])
    x = convBlock(filters=128)(x)

    x = Conv2DTranspose(filters=128,kernel_size=2,strides=2,
                                                  use_bias=False,
                                                  kernel_initializer=kernel_initializer(seed=27))(x)
    x = concatenate([x,X[3]])
    x = convBlock(filters=64)(x)

    x = Conv2DTranspose(filters=64,kernel_size=2,strides=2,
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer(seed=10))(x)
    x = concatenate([x,X[4]])
    x = convBlock(filters=64)(x)

    x = Conv2D(filters=1,kernel_size=3,padding='same',activation='sigmoid',
                                                          use_bias=False,
                                                          kernel_initializer=kernel_initializer(seed=23))(x)
    #x = Softmax()(x)
    return Model(input_,x,name=name)

def get_model(name='UNet2D'):
    x = input_ = Input(shape=(None,None,3))
    X = go_down()(x)
    x = go_up()(X)
    return Model(input_,x,name=name)

def get_model_transfer(name='UNet2D',vgg_path='./model/models/vgg11'):
    model = get_model(name=name)
    transfer(model,vgg_path=vgg_path)
    return model 


#def load_model(name='name',dir_model='./model/models/'):
#    latest = sorted(glob(f'{dir_model}model_{name}*'))[-1]
#    model = tf.keras.models.load_model(latest)
#    return model

def load_model(name='name',dir_model='./model/models/'):
    dir_weights = f'{dir_model}{name}/'
    model = get_model(name=name)
    latest = tf.train.latest_checkpoint(dir_weights)
    model.load_weights(latest).expect_partial()
    return model


def load_models():
    model_sagital = load_model(name='sagital')
    model_coronal = load_model(name='coronal')
    model_axial = load_model(name='axial')
    return model_sagital,model_coronal,model_axial


if __name__ == "__main__":
    model_sagital = get_model_transfer('unet')
    model_sagital.summary()
    model_sagital.save_weights('./weights/weights.cpkt')
    model_sagital.load_weights('./model/weights_sagital/cp-0001.ckpt')

    model_coronal = get_model_transfer('unet')
    model_coronal.summary()
    model_coronal.save_weights('./weights/weights.cpkt')
    model_coronal.load_weights('./weights/weights.cpkt')

    model_axial = get_model_transfer('unet')
    model_axial.summary()
    model_axial.save_weights('./weights/weights.cpkt')
    model_axial.load_weights('./weights/weights.cpkt')

    matrix = tf.zeros((1,160,160,3))
    vol = model_sagital.predict(matrix)
    print(print(vol.shape))

