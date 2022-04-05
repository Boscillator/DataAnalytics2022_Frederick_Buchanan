from tkinter import W
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from FirePrediction.data_loader import FireSequence
from FirePrediction.train_cnn import weighted_bincrossentropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=pathlib.Path)
    args = parser.parse_args()

    model = keras.models.load_model(args.model, custom_objects={'weighted_bincrossentropy': weighted_bincrossentropy})

    test_generator = FireSequence('data/processed/test', 10)

    evaluation = model.evaluate(test_generator, verbose=1)
    for name, value in zip(model.metrics_names, evaluation):
        print(f"{name}: {value}")

    if not os.path.exists("plots/evaluation"):
        os.makedirs("plots/evaluation")

    for i in range(len(test_generator)):
        Xs, ys = test_generator[i]
        ys_pred = model.predict_on_batch(Xs)
        meta = test_generator.meta_for(i)

        for j in range(5):
            xs_ = Xs[j,:,:,0]
            ys_ = ys[j,:,:]
            ys_pred_ = ys_pred[j,:,:,0] > 0.5
            plot_prediction(xs_, ys_, ys_pred_, ys_pred[j,:,:,0], meta[j])

def plot_prediction(xs, ys, ys_pred, full_predictions, meta):
    true_positives = np.logical_and(ys, ys_pred)
    false_positives = np.logical_and(np.logical_not(ys), ys_pred)
    true_negatives = np.logical_and(np.logical_not(ys), np.logical_not(ys_pred))
    false_negatives = np.logical_and(ys, np.logical_not(ys_pred))

    spread = ys - xs
    true_positive_spread = np.logical_and(true_positives, spread)
    false_positive_spread = np.logical_and(false_positives, spread)

    plt.clf()
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(13.333, 7.5)

    axs = axs.reshape(-1)

    axs[0].imshow(ys)
    axs[0].set_title("Ground truth")

    axs[1].imshow(full_predictions)
    axs[1].set_title("Prediction")

    axs[2].imshow(true_positives)
    axs[2].set_title("True positives")

    axs[3].imshow(false_positives)
    axs[3].set_title("False positives")

    axs[4].imshow(true_negatives)
    axs[4].set_title("True negatives")

    axs[5].imshow(false_negatives)
    axs[5].set_title("False negatives")

    axs[6].imshow(spread)
    axs[6].set_title("Spread")

    axs[7].imshow(true_positive_spread)
    axs[7].set_title("True positive spread")

    axs[8].imshow(false_positive_spread)
    axs[8].set_title("False positive spread")

    plt.show()


def old_style_plots(model, i, Xs, ys, ys_pred, meta):
    for j in range(5):
        plt.clf()
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(13.333, 7.5)

        axs = axs.reshape(-1)
 
        axs[0].imshow(ys[j,:,:], vmin=0, vmax=1)
        axs[0].set_title("t+1 true (where it is tomorrow)")

        axs[1].imshow(ys_pred[j,:,:,0], vmin=0, vmax=1)
        axs[1].set_title("t+1 predicted (where we think it will be tomorrow)")

        axs[2].imshow(np.logical_xor(ys[j,:,:], ys_pred[j,:,:,0] > 0.8), vmin=0, vmax=1)
        axs[2].set_title("true - predicted (what we got wrong)")

        axs[3].imshow(Xs[j,:,:,0])
        axs[3].set_title("t true (where it was yesterday)")

        axs[4].imshow(ys[j,:,:] - Xs[j,:,:,0])
        axs[4].set_title("(t + 1 true) - t (how it actually spread)")

        axs[5].imshow((ys_pred[j,:,:,0] > 0.8) - Xs[j,:,:,0], vmin=0, vmax=1)
        axs[5].set_title("(t+1 predicted) - (t), (how we thought it would spread)")

        fig.suptitle(meta[j].file_name)
        plt.tight_layout()
        plt.show()
        fig.savefig(f"plots/evaluation/{model.name}_{i}_{j}.png")





if __name__ == '__main__':
    main()