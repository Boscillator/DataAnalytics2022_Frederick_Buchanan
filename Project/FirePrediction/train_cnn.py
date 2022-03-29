import pathlib
import numpy as np
import tensorflow as tf
from keras import layers, models, initializers, losses, metrics
import keras
from FirePrediction.data_loader import FireSequence
import matplotlib.pyplot as plt

BATCH_SIZE = 50

def build_model(size, input_layers):
    model = models.Sequential()
    model.add(layers.Input((size,size,input_layers)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(layers.Conv2D(1, (5,5), padding='same', activation='relu'))
    model.add(layers.Activation('sigmoid'))
    return model

def cust_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)

    y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
    t = tf.cast(y_pred == y_true, dtype=tf.float32)
    return tf.reduce_sum(t) / tf.cast(tf.size(t), dtype=tf.float32)

def visualize(model):
    training_generator = FireSequence(pathlib.Path('data/processed/test'), 5)
    Xs, ys = training_generator[0]
    ys_pred = model.predict_on_batch(Xs)
    for b in range(ys.shape[0]):
        y = ys[b,:].reshape(64,64)
        x = Xs[b,:,:,0]
        y_pred = ys_pred[b,:].reshape(64,64)
        f, axs = plt.subplots(1,3)
        axs[0].imshow(x)
        axs[0].set_title("Source")
        axs[1].imshow(y)
        axs[1].set_title("Truth")
        axs[2].imshow(y_pred)
        axs[2].set_title("Predicted")
        plt.show()


def weighted_bincrossentropy(true, pred, weight_zero = 0.003, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.

    From: https://github.com/huanglau/Keras-Weighted-Binary-Cross-Entropy/blob/master/DynCrossEntropy.py
    """

    # true = tf.cast(true, dtype=tf.float32)
    # pred = tf.cast(pred, dtype=tf.float32)
  
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)

def main():
    model = build_model(64, 5)
    training_generator = FireSequence(pathlib.Path('data/processed/train'), BATCH_SIZE)
    validation_generator = FireSequence(pathlib.Path('data/processed/validate'), BATCH_SIZE)

    model.compile(
        optimizer='adam',
        loss=weighted_bincrossentropy,
        metrics=['accuracy', metrics.Precision(thresholds=0.5), metrics.Recall(thresholds=0.5)]
    )

    print(model.summary())
    model.fit(training_generator, epochs=10, validation_data=validation_generator)

    model.save('data/model')


    # weights = model.layers[-3].get_weights()
    # print(weights)
    visualize(model)

if __name__ == '__main__':
    main()
