from tkinter import W
import keras
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from FirePrediction.data_loader import FireSequence
from FirePrediction.train_cnn import weighted_bincrossentropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    model = keras.models.load_model(args.model, custom_objects={'weighted_bincrossentropy': weighted_bincrossentropy})

    test_generator = FireSequence('data/processed/test', 10)

    if not os.path.exists("plots/evaluation"):
        os.makedirs("plots/evaluation")

    for i in range(len(test_generator)):
        Xs, ys = test_generator[i]
        ys_pred = model.predict_on_batch(Xs)
        meta = test_generator.meta_for(i)


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
            fig.savefig(f"plots/evaluation/{i}_{j}.png")





if __name__ == '__main__':
    main()