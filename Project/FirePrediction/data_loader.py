import pathlib
import rasterio
import random
import numpy as np
import math
from tensorflow.keras.utils import Sequence
from FirePrediction.util import iterate_folder
from pprint import pprint


class FireSequence(Sequence):

    def __init__(self, folder: pathlib.Path, batch_size: int, differential=False):
        self.files = list(iterate_folder(folder))
        random.shuffle(self.files)
        self.batch_size = batch_size
        self.differential = differential

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.files)

    def meta_for(self, idx):
        files = self.files[idx*self.batch_size:(idx+1)*self.batch_size]
        return files

    def __getitem__(self, idx):
        files = self.files[idx*self.batch_size:(idx+1)*self.batch_size]

        ys = []
        Xs = []

        for f in files:
            raster = rasterio.open(f.path)
            y = raster.read(1).astype(np.float32)

            if self.differential:
                input = raster.read(2)
                y = np.logical_xor(y, input).astype(np.float32)

            ys.append(y)

            layers = []
            for layer in raster.indexes[1:]:
                layers.append(raster.read(layer).astype(np.float32))

            Xs.append(np.stack(layers))

        ys = np.stack(ys)
        Xs = np.stack(Xs)
        Xs = np.moveaxis(Xs, 1, -1)
        return Xs, ys


if __name__ == '__main__':
    loader = FireSequence(pathlib.Path('data/processed/train'), 20)
    print(len(loader))
    Xs, ys = loader[0]

    import matplotlib.pyplot as plt
    f, axs = plt.subplots(1,2)
    axs[0].imshow(Xs[0, :, :, 0])
    axs[1].imshow(ys[0,:].reshape(64,64))
    plt.show()
