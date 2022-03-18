import matplotlib.pyplot as plt
import rasterio
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--band', type=int, default=None)
    args = parser.parse_args()

    f = rasterio.open(args.file)

    if args.band is None:
        bands = list(f.indexes)
    else:
        bands  = [args.band]

    for band in bands:
        print("Band")
        plt.gcf()
        data = f.read(band)
        plt.imshow(data)
        plt.title(f"Band: {band:02d}")
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    main()
