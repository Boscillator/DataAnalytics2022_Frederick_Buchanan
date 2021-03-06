from gc import callbacks
import pathlib
from turtle import st
import numpy as np
import tensorflow as tf
from keras import layers, models, initializers, losses, metrics, callbacks
import keras
from FirePrediction.data_loader import FireSequence
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 100

def build_discriminator(size, input_layers):
    # Taken from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
    in_src_image = layers.Input(shape=(size,size,input_layers))
    in_target_image = layers.Input(shape=(size,size))

    merged = layers.concatenate([in_src_image, in_target_image])

    d = layers.Conv2D(64, (4,4), stride=(2,2), padding='same')(merged)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(128, (4,4), stride=(2,2), padding='same')(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(512, (4,4), strides=(2,2), padding='same')(d)
    

def build_unet(size, input_layers, start_neurons=16):
    # Taken from: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    input_layer = layers.Input((size, size, input_layers))
    batch_norm = layers.BatchNormalization()(input_layer)

    conv1 = layers.Conv2D(start_neurons*1, (3, 3),
                          activation='relu', padding='same')(batch_norm)
    conv1 = layers.Conv2D(start_neurons*1, (3, 3),
                          activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(start_neurons*2, (3, 3),
                          activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(start_neurons*2, (3, 3),
                          activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(start_neurons*4, (3, 3),
                          activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(start_neurons*4, (3, 3),
                          activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPool2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(start_neurons*8, (3, 3),
                          activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(start_neurons*8, (3, 3),
                          activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    # Middle
    convm = layers.Conv2D(start_neurons*16, (3, 3),
                          activation='relu', padding='same')(pool4)
    convm = layers.Conv2D(start_neurons*16, (3, 3),
                          activation='relu', padding='same')(convm)

    # Deconvolution
    deconv4 = layers.Conv2DTranspose(
        start_neurons*8, (3, 3), strides=(2, 2), padding='same')(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(start_neurons*8, (3, 3),
                           activation='relu', padding='same')(uconv4)
    uconv4 = layers.Conv2D(start_neurons*8, (3, 3),
                           activation='relu', padding='same')(uconv4)

    deconv3 = layers.Conv2DTranspose(
        start_neurons*4, (3, 3), strides=(2, 2), padding='same')(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(start_neurons*4, (3, 3),
                           activation='relu', padding='same')(uconv3)
    uconv3 = layers.Conv2D(start_neurons*4, (3, 3),
                           activation='relu', padding='same')(uconv3)

    deconv2 = layers.Conv2DTranspose(
        start_neurons*2, (3, 3), strides=(2, 2), padding='same')(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(start_neurons*2, (3, 3),
                           activation='relu', padding='same')(uconv2)
    uconv2 = layers.Conv2D(start_neurons*2, (3, 3),
                           activation='relu', padding='same')(uconv2)

    deconv1 = layers.Conv2DTranspose(
        start_neurons*1, (3, 3), strides=(2, 2), padding='same')(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(start_neurons*1, (3, 3),
                           activation='relu', padding='same')(uconv1)
    uconv1 = layers.Conv2D(start_neurons*1, (3, 3),
                           activation='relu', padding='same')(uconv1)

    output_layer = layers.Conv2D(
        1, (1, 1), padding='same', activation='sigmoid')(uconv1)

    return keras.Model(inputs=input_layer, outputs=output_layer, name='fire_unet')


def weighted_bincrossentropy(true, pred, weight_zero=0.01, weight_one=1):
    """
    From: https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
    """
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)


def main():
    model = build_model(64, 6)
    training_generator = FireSequence(
        pathlib.Path('data/processed/train'), BATCH_SIZE)
    validation_generator = FireSequence(
        pathlib.Path('data/processed/validate'), BATCH_SIZE)

    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy',
                 metrics.Precision(thresholds=0.5),
                 metrics.Recall(thresholds=0.5)]
    )

    print(model.summary())
    model.fit(training_generator, epochs=10, validation_data=validation_generator)

    if not os.path.exists('data/models'):
        os.makedirs('data/models')

    model.save('data/models/pix2pix')


if __name__ == '__main__':
    main()
