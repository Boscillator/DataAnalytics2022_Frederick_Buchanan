import dataclasses
import pathlib
import os
from typing import Iterator, Union


@dataclasses.dataclass(frozen=True)
class DataPoint:
    year: int
    month: int
    sample_no: int
    ordinal_day: int
    file_name: str
    path: pathlib.Path


def iterate_folder(path: Union[str, pathlib.Path]) -> Iterator[DataPoint]:
    path = pathlib.Path(path)
    files = os.listdir(path)
    for name in files:
        year, month, sample_no, ordinal_day, _ = name.split('.')
        yield DataPoint(
            year=int(year),
            month=int(month),
            sample_no=int(sample_no),
            ordinal_day=int(ordinal_day),
            file_name=name,
            path=path / name
        )


if __name__ == '__main__':
    for item in iterate_folder(pathlib.Path('data/processed/cropped_and_sampled')):
        print(item)
