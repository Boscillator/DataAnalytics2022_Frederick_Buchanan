import datetime
import pathlib
import rasterio
from pydoc import cli
import paramiko
import os

FTP_HOST = 'fuoco.geog.umd.edu'
FTP_PORT = 22
FTP_USER = 'fire'
FTP_PASS = 'burnt'

WINDOW = 3
LOCAL_DIR = 'data/raw/mcd64a1/'

def make_remote_path(window, year, month):
    ordinal_day = datetime.date(year=year, month=month, day=1).timetuple().tm_yday
    return f'/data/MODIS/C6/MCD64A1/TIFF/Win{window:02d}/{year}/MCD64monthly.A{year}{ordinal_day:03d}.Win{window:02d}.006.burndate.tif'

def make_local_path(year, month):
    return pathlib.Path(LOCAL_DIR) / f'{year}.{month}.tif'

def download_from_server(client, window, year, month):
    remote_path = make_remote_path(window, year, month)
    local_path = make_local_path(year, month)

    if not os.path.exists(local_path.parent):
        os.makedirs(local_path.parent)

    client.get(remote_path, local_path)

def make_client():
    paramiko.util.log_to_file("paramiko.log")
    transport = paramiko.Transport((FTP_HOST, FTP_PORT))
    transport.connect(None, FTP_USER, FTP_PASS)

    return paramiko.SFTPClient.from_transport(transport)

def open_mcd64a1(year: int, month: int):
    path = make_local_path(year, month)
    if not os.path.exists(path):
        client = make_client()
        download_from_server(client, WINDOW, year, month)

    return rasterio.open(path)

if __name__ == '__main__':
    data = open_mcd64a1(2012, 3)
    print(data.bounds)