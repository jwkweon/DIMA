import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

tf.enable_eager_execution()

##################################################################################
# Hyper Parameter
##################################################################################
learning_rate = 0.001
training_epochs = 1000

##################################################################################
# Data Loader
##################################################################################
def dataloader():
    '''dataset의 directory에서 이미지를 불러와 nparray로 반환해주는 func'''
    path_dir = os.getcwd()
    path_data = path_dir + '/dataset/'
    data_list = os.listdir(path_data)

    for i, image_name in enumerate(data_list):
        image = np.array(Image.open(path_data+image_name))
        image = np.expand_dims(image, axis = 0)
        if image.shape[-1] != 3:
            continue
        if i == 0:
            image_ex = image
        else:
            image_ex = np.concatenate((image_ex, image), axis=0)
        print(image_ex.shape)
    return image_ex

##################################################################################
# Model Structure
##################################################################################
def create_model():
    '''Camo_gen network'''
    inputs_S = keras.Input(shape=(128, 128, 3))
    inputs_P = keras.Input(shape=(128, 128, 3))

    ####### Scenery Feature Extract
    conv1_S = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(inputs_S)
    pool1_S = keras.layers.MaxPool2D(padding = 'SAME')(conv1_S)
    conv2_S = keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(pool1_S)
    pool2_S = keras.layers.MaxPool2D(padding = 'SAME')(conv2_S)
    conv3_S = keras.layers.Conv2D(filters=256, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(pool2_S)
    pool3_S = keras.layers.MaxPool2D(padding = 'SAME')(conv3_S)

    ####### Pattern Feature Extract - Encoder
    conv1_P = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(inputs_P)
    pool1_P = keras.layers.MaxPool2D(padding = 'SAME')(conv1_P)
    conv2_P = keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(pool1_P)
    pool2_P = keras.layers.MaxPool2D(padding = 'SAME')(conv2_P)
    conv3_P = keras.layers.Conv2D(filters=256, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(pool2_P)
    pool3_P = keras.layers.MaxPool2D(padding = 'SAME')(conv3_P)
    ####### Pattern Feature Extract - Concat Scenery Feauter & Pattern Feature
    concat1 = keras.layers.Concatenate(axis=-1)([pool3_P, pool3_S])
    ####### Pattern Feature Extract - Decoder
    upsamp1 = keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='SAME',
                               activation=keras.layers.ReLU())(concat1)
    concat2 = keras.layers.Concatenate(axis=-1)([upsamp1, conv3_P])
    conv4_P = keras.layers.Conv2D(filters=256, kernel_size=1, padding='SAME',
                               activation=keras.layers.ReLU())(concat2)
    upsamp2 = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='SAME',
                               activation=keras.layers.ReLU())(conv4_P)
    concat3 = keras.layers.Concatenate(axis=-1)([upsamp2, conv2_P])
    conv5_P = keras.layers.Conv2D(filters=128, kernel_size=1, padding='SAME',
                               activation=keras.layers.ReLU())(concat3)
    upsamp3 = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='SAME',
                               activation=keras.layers.ReLU())(conv5_P)
    concat4 = keras.layers.Concatenate(axis=-1)([upsamp3, conv1_P])


    conv5_P = keras.layers.Conv2D(filters=64, kernel_size=1, padding='SAME',
                               activation=keras.layers.ReLU())(concat4)
    conv6_P = keras.layers.Conv2D(filters=3, kernel_size=3, padding='SAME',
                               activation=keras.layers.ReLU())(conv5_P)

    model = keras.Model(inputs=(inputs_S,inputs_P) , outputs=conv6_P)
    return model


##################################################################################
# Loss Function
##################################################################################
def loss_fn(model, images, images_pattern):
    '''우선 loss_pattern 만 갖고 학습'''
    #training = True를 하면 model 정의 부분의 dropout 이 적용됨! 나머지는 영향 없음
    G = model_test(images, images_pattern, training = True)

    loss_pattern = tf.reduce_mean(tf.squared_difference(G, images_pattern))
    ##################################################################################
    # 이 부분 채우기
    #loss_color =
    ##################################################################################

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return loss_pattern

def main():


if __name__ == '__main__':
    main()
