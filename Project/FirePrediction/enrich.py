import argparse
import multiprocessing
import pathlib
import os
import rasterio
from FirePrediction.util import iterate_folder, DataPoint
from FirePrediction.uvwnd import open_component
from FirePrediction.airtemp import open_air_temperature



def enrich(image: DataPoint, output_folder: pathlib.Path):
    print(f"Enriching: {image.file_name}")
    sample = rasterio.open(image.path)
    sample_shape = sample.read(1).shape
    profile = sample.profile

    wind_u = open_component(image.year, image.ordinal_day,
                            'u', sample.bounds, sample_shape)
    wind_v = open_component(image.year, image.ordinal_day,
                            'v', sample.bounds, sample_shape)
    air = open_air_temperature(
        image.year, image.ordinal_day, sample.bounds, sample_shape)

    profile.update({
        'count': 6
    })

    with rasterio.open(output_folder / image.file_name, 'w', **profile) as dst:
        dst.write(sample.read(1), 1)    # Prediction
        dst.write(sample.read(2), 2)    # Fire Mask
        dst.write(sample.read(3), 3)    # Inverse Duration
        dst.write(wind_u, 4)
        dst.write(wind_v, 5)
        dst.write(air, 6)


def main():
    parser = argparse.ArgumentParser(description="Add exogenous data")
    parser.add_argument("--input", type=pathlib.Path, default=pathlib.Path("data/processed/cropped_and_sampled"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/processed/enriched"))
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i, image in enumerate(iterate_folder(args.input)):
        print(i)
        enrich(image, args.output)

if __name__ == '__main__':
    main()
