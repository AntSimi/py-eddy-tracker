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
from os import path
import tarfile

import requests


def get_demo_path(name):
    return path.join(path.dirname(__file__), name)


def get_remote_demo_sample(path):
    if path.startswith("/") or path.startswith("."):
        content = open(path, "rb").read()
        if path.endswith(".nc"):
            return io.BytesIO(content)
    else:
        if path.endswith(".nc"):
            content = requests.get(
                f"https://github.com/AntSimi/py-eddy-tracker-sample-id/raw/master/{path}"
            ).content
            return io.BytesIO(content)
        content = requests.get(
            f"https://github.com/AntSimi/py-eddy-tracker-sample-id/raw/master/{path}.tar.xz"
        ).content

    # Tar module could manage lzma tar, but it will apply uncompress for each extractfile
    tar = tarfile.open(mode="r", fileobj=io.BytesIO(lzma.decompress(content)))
    # tar = tarfile.open(mode="r:xz", fileobj=io.BytesIO(content))
    files_content = list()
    for item in tar:
        content = tar.extractfile(item)
        content.filename = item.name
        files_content.append(content)
    return files_content
