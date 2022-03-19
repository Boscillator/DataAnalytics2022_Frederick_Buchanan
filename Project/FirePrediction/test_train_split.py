import argparse
import os
import pathlib
import shutil
from FirePrediction.util import iterate_folder

def create_if_not_exists(path: pathlib.Path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, default=pathlib.Path('data/processed/enriched'))
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('data/processed'))
    parser.add_argument('--n', type=int, default=5)
    args = parser.parse_args()

    train = args.output / 'train'
    test = args.output / 'test'
    validate = args.output / 'validate'

    create_if_not_exists(train)
    create_if_not_exists(test)
    create_if_not_exists(validate)

    for image in iterate_folder(args.input):
        print(image.file_name)
        if image.sample_no % args.n == 0:
            shutil.copy(image.path, test / image.file_name)
        if image.sample_no % args.n == 1:
            shutil.copy(image.path, validate / image.file_name)
        else:
            shutil.copy(image.path, train / image.file_name)

if __name__ == '__main__':
    main()
