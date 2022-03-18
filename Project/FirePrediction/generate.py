import rasterio
from FirePrediction.mcd64a1 import open_mcd64a1
import argparse
import numpy as np
import os
import calendar
import datetime

def main():
    args = parse_args()
    raster = open_mcd64a1(args.year, args.month)
    sample = sample_fire_pixels(raster, args.n)

    if not os.path.exists('data/processed/cropped_and_sampled'):
        os.makedirs('data/processed/cropped_and_sampled')

    for n in range(args.n):
        print(f'{n + 1}/{args.n}')

        x, y = sample[n, :]
        window, profile = crop(n, args.month, args.year, raster, x, y, args.size)
        fireband = raster.read(1, window=window)
        for ordinal in get_range_from_sample(fireband):
            active_fire = fireband >= ordinal
            one_over = one_over_distance(fireband, ordinal)

            with rasterio.open(f'data/processed/cropped_and_sampled/{args.year}.{args.month:02d}.{n:02d}.{ordinal:03d}.tif', 'w', **profile) as dst:
                dst.write(active_fire, 1)
                dst.write(one_over * 1024, 2)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a dataset for a given month and year')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()


def has_burned_this_month_mask(data):
    fireband = data.read(1)
    return fireband > 0


def sample_fire_pixels(data, n):
    burn_mask = has_burned_this_month_mask(data)
    burned_pixels = np.where(burn_mask)
    burned_pixels = np.stack(burned_pixels, axis=1)

    sample_indicies = np.random.randint(0, burned_pixels.shape[0], n)
    return burned_pixels[sample_indicies, :]


def crop(n, month, year, raster, x, y, size):
    x -= size//2
    y -= size//2
    window = rasterio.windows.Window(y, x, size, size)

    transform = raster.window_transform(window)

    profile = raster.profile
    profile.update({
        'height': size,
        'width': size,
        'transform': transform,
        'count': 2
    })

    return window, profile

def get_range_from_sample(data):
    min = np.min(data[data > 0])
    max = np.max(data[data > 0])
    return range(min, max+1)

def one_over_distance(fireband, ordinal):
    fireband = fireband.copy()
    fireband -= ordinal
    fireband[fireband <= 0] = 0
    fireband = 1/fireband
    return np.nan_to_num(fireband)

if __name__ == '__main__':
    main()
