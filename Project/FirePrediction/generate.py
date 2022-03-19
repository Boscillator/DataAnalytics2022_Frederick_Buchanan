import rasterio
from FirePrediction.mcd64a1 import open_mcd64a1
import argparse
import numpy as np
import os

def main():
    args = parse_args()
    np.random.seed(args.seed)

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
            active_fire = (fireband <= ordinal) & (fireband > 0)
            active_tommorrow = (fireband <= ordinal + 1) & (fireband > 0)
            one_over = one_over_distance(fireband, ordinal)

            with rasterio.open(f'data/processed/cropped_and_sampled/{args.year}.{args.month:02d}.{n:02d}.{ordinal:03d}.tif', 'w', **profile) as dst:
                dst.write(active_tommorrow, 1)
                dst.write(active_fire, 2)
                dst.write(one_over * 1024, 3)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a dataset for a given month and year')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
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
        'count': 3
    })

    return window, profile

def get_range_from_sample(data):
    mmin = np.min(data[data > 0])
    mmax = np.max(data[data > 0])
    return range(mmin, mmax+1)

def one_over_distance(fireband_original, ordinal):
    fireband = fireband_original.copy()

    fireband = ordinal - fireband
    fireband = 1/fireband

    fireband[fireband_original > ordinal] = 0
    fireband[fireband_original <= 0] = 0

    fireband = np.nan_to_num(fireband, nan=0.0, posinf=0.0, neginf=0.0)
    return fireband

if __name__ == '__main__':
    main()
