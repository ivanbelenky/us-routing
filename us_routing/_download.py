import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger('us_routing')

URL_GETTER_URL = 'https://hub.arcgis.com/api/download/v1/items/0b6c2fd2e3ac40a7929cdff1d4cf604a/{format}?redirect=false&layers=0'

US_ROUTING_HOME = Path.home()/".us_routing"
ONLY_HIGHWAYS_GRAPH_FILE_PATH = US_ROUTING_HOME/"usrouter1.pkl"
HIGHWAYS_AND_PRINCIPAL_ROADS_GRAPH_FILE_PATH = US_ROUTING_HOME/"usrouter12.pkl"
HIGHWAYS_PRINCIPALS_AND_SECONDARY_ROADS_GRAPH_FILE_PATH = US_ROUTING_HOME/"usrouter123.pkl"


ONLY_HIGHWAYS_GRAPH_FILE_DOWNLOAD_URL = "https://github.com/ivanbelenky/us-routing/releases/download/0.1.0/usrouting1.pkl"
HIGHWAYS_AND_PRINCIPAL_ROADS_GRAPH_FILE_DOWNLOAD_URL = "https://github.com/ivanbelenky/us-routing/releases/download/0.1.0/usrouting12.pkl"
HIGHWAYS_PRINCIPALS_AND_SECONDARY_ROADS_GRAPH_FILE_DOWNLOAD_URL = "https://github.com/ivanbelenky/us-routing/releases/download/0.1.0/usrouting123.pkl"

NAR_PATH = US_ROUTING_HOME/"north_american_roads"


FMT_MAPPER = {
    'sqlite': 'sqlite',
    'zip': 'file-geodatabase',
    'shp': 'shapefile'
}

def _get_url(url=URL_GETTER_URL):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['resultUrl']


def _download_north_american_roads(url, fmt, path=NAR_PATH):
    match fmt:
        case 'sqlite': name = 'north_american_roads.sqlite'
        case 'shp': name = 'north_american_roads.shp'
        case _: raise ValueError(f"Unknown extension: {fmt}")

    file_path = Path(path, name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        logger.info(f"File already exists: {file_path}")
        return

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
        logger.error("ERROR, something went wrong")

    if fmt == 'zip':
        with zipfile.ZipFile(path/name, 'r') as zip_ref:
            zip_ref.extractall(path)
        (path/name).unlink()


def download_north_american_roads(path: Path=NAR_PATH, fmt: str='shp'):
    if fmt not in ['sqlite', 'shp']:
        raise ValueError("data_files must be a list of ['sqlite', 'shp']")

    url = _get_url(URL_GETTER_URL.format(format=FMT_MAPPER[fmt]))
    _download_north_american_roads(url, fmt, path)


def download_graph_if_not_exists(
    db_file_path: Path | str,
    download_url: str,
):
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)
    if db_file_path.exists(): return

    response = requests.get(download_url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 4096
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(db_file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("ERROR, something went wrong")
