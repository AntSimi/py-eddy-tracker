"""
EddyId \
    nrt_global_allsat_phy_l4_20190223_20190226.nc \
    20190223 adt ugos vgos longitude latitude . \
    --cut 800 --fil 1
EddyId \
    dt_med_allsat_phy_l4_20160515_20190101.nc \
    20160515 adt None None longitude latitude . \
    --cut 800 --fil 1
"""
import io
import lzma
import tarfile
from os import path

import requests


def get_path(name):
    return path.join(path.dirname(__file__), name)


def get_remote_sample(path):
    url = (
        f"https://github.com/AntSimi/py-eddy-tracker-sample-id/raw/master/{path}.tar.xz"
    )

    content = requests.get(url).content

    # Tar module could manage lzma tar, but it will apply un compress for each extractfile
    tar = tarfile.open(mode="r", fileobj=io.BytesIO(lzma.decompress(content)))
    # tar = tarfile.open(mode="r:xz", fileobj=io.BytesIO(content))
    files_content = list()
    for item in tar:
        content = tar.extractfile(item)
        content.filename = item.name
        files_content.append(content)
    return files_content
