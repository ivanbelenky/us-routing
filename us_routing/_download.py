import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

NAR_PATH = Path("./assets/")
URL_GETTER_URL = 'https://hub.arcgis.com/api/download/v1/items/0b6c2fd2e3ac40a7929cdff1d4cf604a/{format}?redirect=false&layers=0'


def _get_url(url=URL_GETTER_URL):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['resultUrl']


def _download_north_american_roads(url, path=NAR_PATH):
    extension = url.split('.')[-1]
    match extension:
        case 'sqlite': name = 'north_american_roads.sqlite'
        case 'zip': name = 'north_american_roads.zip'
        case _: raise ValueError(f"Unknown extension: {extension}")

    if not path.exists(): path.mkdir()
    if (path/'north_american_roads.sqlite').exists(): return

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 4096
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(path/name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    if extension == 'zip':
        with zipfile.ZipFile(path/name, 'r') as zip_ref:
            zip_ref.extractall(path)
        (path/name).unlink()


def download_north_american_roads(fmt: str=None):
    if fmt not in ['sqlite', 'zip', 'gdb', 'shp']:
        raise ValueError("data_files must be a list of ['sqlite', 'zip', 'gdb', 'shp']")

    url = _get_url(URL_GETTER_URL.format(format=fmt))
    _download_north_american_roads(url)