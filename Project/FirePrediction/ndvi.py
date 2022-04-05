from datetime import datetime
import numpy as np
import urllib
from pyrsistent import get_in
import rasterio
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from bs4 import BeautifulSoup
from functools import lru_cache
import urllib.request


LOCAL_DIR = 'data/raw/ndvi/'

INDEX_URL = 'https://www.ncei.noaa.gov/data/avhrr-land-normalized-difference-vegetation-index/access'


@lru_cache
def get_index(year):
    fp = urllib.request.urlopen(f'{INDEX_URL}/{year}')
    html = fp.read().decode('utf-8')
    fp.close()
    soup = BeautifulSoup(html, features="lxml")
    return soup


def jd_to_date(year, oridnal):
    fmt = "%Y%j"
    datestd = datetime.strptime(f"{year}{oridnal}", fmt).date()
    return datestd


def make_url(year, ordinal):
    index = get_index(year)
    date = jd_to_date(year, ordinal).strftime("%Y%m%d")

    for link in index.find_all('a'):
        if date in link.text:
            return f"{INDEX_URL}/{year}/{link.get('href')}"


def make_local_path(year, ordinal):
    return f"{LOCAL_DIR}{year}.{ordinal}.nc"


def download_from_server(year, ordinal):
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    url = make_url(year, ordinal)
    local = make_local_path(year, ordinal)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, local)


def open_ndvi(year, ordinal, bounds, out_shape):
    local = make_local_path(year, ordinal)
    if not os.path.exists(local):
        download_from_server(year, ordinal)

    dataset = rasterio.open(f"netcdf:{local}:NDVI")
    window = rasterio.windows.from_bounds(
        top=bounds.top,
        bottom=bounds.bottom,
        right=bounds.right,
        left=bounds.left,
        transform=dataset.transform
    )

    img = dataset.read(1, window=window, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)


    if np.all(img == -9999):
        return None

    return img


if __name__ == '__main__':
    fire = rasterio.open('data/processed/cropped_and_sampled/2020.06.06.175.tif')
    n = open_ndvi(2020, 1, fire.bounds, fire.shape)

    plt.imshow(n)
    plt.show()
