from functools import reduce
import matplotlib.pyplot as plt
import rasterio
import argparse
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--band', type=int, default=None)
    parser.add_argument('--tile', default=False, action='store_true')
    args = parser.parse_args()

    f = rasterio.open(args.file)

    if args.band is None:
        bands = list(f.indexes)
    else:
        bands  = [args.band]

    if not args.tile:
        sequence(f, bands)
    else:
        tile(f, bands)


def sequence(f, bands):
    for band in bands:
        print("Band")
        plt.gcf()
        data = f.read(band)
        plt.imshow(data)
        plt.title(f"Band: {band:02d}")
        plt.colorbar()
        plt.show()

def tile(f, bands):
    columns = 2
    rows = math.ceil(len(bands)/2)
    fig, axs = plt.subplots(rows, columns)
    axs = axs.reshape(-1)

    for band, ax in zip(bands, axs):
        print(band, ax)
        ax.imshow(f.read(band))
        ax.set_title(f"band: {band}")
    
    plt.show()

if __name__ == '__main__':
    main()
