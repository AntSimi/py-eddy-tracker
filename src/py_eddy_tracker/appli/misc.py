# -*- coding: utf-8 -*-
"""
Entry point with no direct link with eddies
"""
import argparse

import zarr


def zarr_header_parser():
    parser = argparse.ArgumentParser("Zarr header")
    parser.add_argument("dataset")
    return parser


def zarrdump():
    args = zarr_header_parser().parse_args()
    print(args.dataset)
    for v in zarr.open(args.dataset).values():
        print(v.info)
