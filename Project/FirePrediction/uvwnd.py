import ftplib
import pathlib
import rasterio
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample

FTP_HOST = 'ftp.cdc.noaa.gov'

LOCAL_DIR = 'data/raw/uvwind/'


def make_remote_path(year, component):
    assert component == 'u' or component == 'v'
    return f'/Projects/Datasets/ncep.reanalysis.dailyavgs/surface/{component}wnd.sig995.{year}.nc'


def make_local_path(year, component):
    assert component == 'u' or component == 'v'
    return pathlib.Path(LOCAL_DIR) / f'{year}.{component}.nc'


def download_from_server(year, component):
    remote = make_remote_path(year, component)
    local = make_local_path(year, component)

    print(f"DOWNLOADING: ftp://{FTP_HOST}/{remote}")

    if not os.path.exists(local.parent):
        os.makedirs(local.parent)

    ftp = ftplib.FTP(FTP_HOST)
    ftp.login('', '')

    with open(local, 'wb') as f:
        ftp.retrbinary("RETR " + remote, f.write)


def c180_to_360(d):
    return (((d - 180) % 360) + 180)


def open_component(year, ordinal, component, bounds, out_shape):
    assert component == 'u' or component == 'v'

    local = make_local_path(year, component)
    if not os.path.exists(local):
        download_from_server(year, component)

    dataset = rasterio.open(f"netcdf:{local}:{component}wnd")
    window = rasterio.windows.from_bounds(
        top=bounds.top,
        bottom=bounds.bottom,
        right=c180_to_360(bounds.right),
        left=c180_to_360(bounds.left),
        transform = dataset.transform    
    )

    return dataset.read(ordinal, window=window, out_shape = out_shape, resampling = rasterio.enums.Resampling.bilinear)


if __name__ == '__main__':
    fire = rasterio.open(
        'data/processed/cropped_and_sampled/2016.03.00.067.tif')
    u = open_component(2016, 67, 'u', fire.bounds, (256,256))
    print(u)

    plt.imshow(u)
    plt.title('u')
    plt.show()

    v = open_component(2016, 1, 'v', fire.bounds, (256, 256))
    plt.imshow(v)
    plt.title('v')
    plt.show()
