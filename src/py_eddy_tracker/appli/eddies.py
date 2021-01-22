# -*- coding: utf-8 -*-
"""
Applications on detection and tracking files
"""
import argparse
import logging
from datetime import datetime
from glob import glob
from os import mkdir
from os.path import basename, dirname, exists
from os.path import join as join_path
from re import compile as re_compile

from netCDF4 import Dataset
from numpy import bincount, bytes_, empty, in1d, unique
from yaml import safe_load

from .. import EddyParser
from ..observations.observation import EddiesObservations, reverse_index
from ..observations.tracking import TrackEddiesObservations
from ..tracking import Correspondances

logger = logging.getLogger("pet")


def eddies_add_circle():
    parser = EddyParser("Add or replace contour with radius parameter")
    parser.add_argument("filename", help="all file to merge")
    parser.add_argument("out", help="output file")
    args = parser.parse_args()
    obs = EddiesObservations.load_file(args.filename)
    if obs.track_array_variables == 0:
        obs.track_array_variables = 50
        obs = obs.add_fields(
            array_fields=(
                "contour_lon_e",
                "contour_lat_e",
                "contour_lon_s",
                "contour_lat_s",
            )
        )
    obs.circle_contour()
    obs.write_file(filename=args.out)


def merge_eddies():
    parser = EddyParser("Merge eddies")
    parser.add_argument("filename", nargs="+", help="all file to merge")
    parser.add_argument("out", help="output file")
    parser.add_argument(
        "--add_rotation_variable", help="add rotation variables", action="store_true"
    )
    parser.add_argument(
        "--include_var", nargs="+", type=str, help="use only listed variable"
    )
    parser.memory_arg()
    args = parser.parse_args()

    if args.include_var is None:
        with Dataset(args.filename[0]) as h:
            args.include_var = h.variables.keys()

    obs = list()
    for filename in args.filename:
        e = TrackEddiesObservations.load_file(
            filename, raw_data=True, include_vars=args.include_var
        )
        if args.add_rotation_variable:
            e = e.add_rotation_type()
        obs.append(e)
    obs = TrackEddiesObservations.concatenate(obs)
    obs.write_file(filename=args.out)


def get_frequency_grid():
    parser = EddyParser("Compute eddy frequency")
    parser.add_argument("observations", help="Input observations to compute frequency")
    parser.add_argument("out", help="Grid output file")
    parser.contour_intern_arg()
    parser.add_argument(
        "--xrange", nargs="+", type=float, help="Horizontal range : START,STOP,STEP"
    )
    parser.add_argument(
        "--yrange", nargs="+", type=float, help="Vertical range : START,STOP,STEP"
    )
    args = parser.parse_args()

    if (args.xrange is None or len(args.xrange) not in (3,)) or (
        args.yrange is None or len(args.yrange) not in (3,)
    ):
        raise Exception("Use START/STOP/STEP for --xrange and --yrange")

    var_to_load = ["longitude"]
    var_to_load.extend(EddiesObservations.intern(args.intern, public_label=True))
    e = EddiesObservations.load_file(args.observations, include_vars=var_to_load)

    bins = args.xrange, args.yrange
    g = e.grid_count(bins, intern=args.intern)
    g.write(args.out)


def display_infos():
    parser = EddyParser("Display General inforamtion")
    parser.add_argument(
        "observations", nargs="+", help="Input observations to compute frequency"
    )
    parser.add_argument("--vars", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("llcrnrlon", "llcrnrlat", "urcrnrlon", "urcrnrlat"),
        help="Bounding box",
    )
    args = parser.parse_args()
    if args.vars:
        vars = args.vars
    else:
        vars = [
            "amplitude",
            "speed_radius",
            "speed_area",
            "effective_radius",
            "effective_area",
            "time",
            "latitude",
            "longitude",
        ]
    filenames = args.observations
    filenames.sort()
    for filename in filenames:
        with Dataset(filename) as h:
            track = "track" in h.variables
        print(f"-- {filename} -- ")
        if track:
            vars_ = vars.copy()
            vars_.extend(("track", "observation_number", "observation_flag"))
            e = TrackEddiesObservations.load_file(filename, include_vars=vars_)
        else:
            e = EddiesObservations.load_file(filename, include_vars=vars)
        if args.area is not None:
            area = dict(
                llcrnrlon=args.area[0],
                llcrnrlat=args.area[1],
                urcrnrlon=args.area[2],
                urcrnrlat=args.area[3],
            )
            e = e.extract_with_area(area)
        print(e)


def eddies_tracking():
    parser = EddyParser("Tool to use identification step to compute tracking")
    parser.add_argument("yaml_file", help="Yaml file to configure py-eddy-tracker")
    parser.add_argument("--correspondance_in", help="Filename of saved correspondance")
    parser.add_argument("--correspondance_out", help="Filename to save correspondance")
    parser.add_argument(
        "--save_correspondance_and_stop",
        action="store_true",
        help="Stop tracking after correspondance computation,"
        " merging can be done with EddyFinalTracking",
    )
    parser.add_argument(
        "--zarr", action="store_true", help="Output will be wrote in zarr"
    )
    parser.add_argument("--unraw", action="store_true", help="Load unraw data")
    parser.add_argument(
        "--blank_period",
        type=int,
        default=0,
        help="Nb of detection which will not use at the end of the period",
    )
    parser.memory_arg()
    args = parser.parse_args()

    # Read yaml configuration file
    with open(args.yaml_file, "r") as stream:
        config = safe_load(stream)

    if "CLASS" in config:
        classname = config["CLASS"]["CLASS"]
        obs_class = dict(
            class_method=getattr(
                __import__(config["CLASS"]["MODULE"], globals(), locals(), classname),
                classname,
            ),
            class_kw=config["CLASS"].get("OPTIONS", dict()),
        )
    else:
        obs_class = dict()

    c_in, c_out = args.correspondance_in, args.correspondance_out
    if c_in is None:
        c_in = config["PATHS"].get("CORRESPONDANCES_IN", None)
    y_c_out = config["PATHS"].get(
        "CORRESPONDANCES_OUT", "{path}/{sign_type}_correspondances.nc"
    )
    if c_out is None:
        c_out = y_c_out

    # Create ouput folder if necessary
    save_dir = config["PATHS"].get("SAVE_DIR", None)
    if save_dir is not None and not exists(save_dir):
        mkdir(save_dir)

    track(
        pattern=config["PATHS"]["FILES_PATTERN"],
        output_dir=save_dir,
        c_out=c_out,
        **obs_class,
        virtual=int(config.get("VIRTUAL_LENGTH_MAX", 0)),
        previous_correspondance=c_in,
        memory=args.memory,
        correspondances_only=args.save_correspondance_and_stop,
        raw=not args.unraw,
        zarr=args.zarr,
        nb_obs_min=int(config.get("TRACK_DURATION_MIN", 10)),
        blank_period=args.blank_period,
    )


def browse_dataset_in(
    data_dir,
    files_model,
    date_regexp,
    date_model,
    start_date=None,
    end_date=None,
    sub_sampling_step=1,
    files=None,
):
    pattern_regexp = re_compile(".*/" + date_regexp)
    if files is not None:
        filenames = bytes_(files)
    else:
        full_path = join_path(data_dir, files_model)
        logger.info("Search files : %s", full_path)
        filenames = bytes_(glob(full_path))

    dataset_list = empty(
        len(filenames),
        dtype=[
            ("filename", "S500"),
            ("date", "datetime64[D]"),
        ],
    )
    dataset_list["filename"] = filenames

    logger.info("%s grids available", dataset_list.shape[0])
    mode_attrs = False
    if "(" not in date_regexp:
        logger.debug("Attrs date : %s", date_regexp)
        mode_attrs = date_regexp.strip().split(":")
    else:
        logger.debug("Pattern date : %s", date_regexp)

    for item in dataset_list:
        str_date = None
        if mode_attrs:
            with Dataset(item["filename"].decode("utf-8")) as h:
                if len(mode_attrs) == 1:
                    str_date = getattr(h, mode_attrs[0])
                else:
                    str_date = getattr(h.variables[mode_attrs[0]], mode_attrs[1])
        else:
            result = pattern_regexp.match(str(item["filename"]))
            if result:
                str_date = result.groups()[0]

        if str_date is not None:
            item["date"] = datetime.strptime(str_date, date_model).date()

    dataset_list.sort(order=["date", "filename"])

    steps = unique(dataset_list["date"][1:] - dataset_list["date"][:-1])
    if len(steps) > 1:
        raise Exception("Several days steps in grid dataset %s" % steps)

    if sub_sampling_step != 1:
        logger.info("Grid subsampling %d", sub_sampling_step)
        dataset_list = dataset_list[::sub_sampling_step]

    if start_date is not None or end_date is not None:
        logger.info(
            "Available grid from %s to %s",
            dataset_list[0]["date"],
            dataset_list[-1]["date"],
        )
        logger.info("Filtering grid by time %s, %s", start_date, end_date)
        mask = (dataset_list["date"] >= start_date) * (dataset_list["date"] <= end_date)

        dataset_list = dataset_list[mask]
    return dataset_list


def track(
    pattern,
    output_dir,
    c_out,
    nb_obs_min=10,
    raw=True,
    zarr=False,
    blank_period=0,
    correspondances_only=False,
    **kw_c,
):
    kw = dict(date_regexp=".*_([0-9]*?).[nz].*", date_model="%Y%m%d")
    if isinstance(pattern, list):
        kw.update(dict(data_dir=None, files_model=None, files=pattern))
    else:
        kw.update(dict(data_dir=dirname(pattern), files_model=basename(pattern)))
    datasets = browse_dataset_in(**kw)
    if blank_period > 0:
        datasets = datasets[:-blank_period]
        logger.info("Last %d files will be pop", blank_period)

    if nb_obs_min > len(datasets):
        raise Exception(
            "Input file number (%s) is shorter than TRACK_DURATION_MIN (%s)."
            % (len(datasets), nb_obs_min)
        )

    c = Correspondances(datasets=datasets["filename"], **kw_c)
    c.track()
    logger.info("Track finish")
    t0, t1 = c.period
    kw_save = dict(
        date_start=t0,
        date_stop=t1,
        date_prod=datetime.now(),
        path=output_dir,
        sign_type=c.current_obs.sign_legend,
    )

    c.save(c_out, kw_save)
    if correspondances_only:
        return

    logger.info("Start merging")
    c.prepare_merging()
    logger.info("Longer track saved have %d obs", c.nb_obs_by_tracks.max())
    logger.info(
        "The mean length is %d observations for all tracks", c.nb_obs_by_tracks.mean()
    )

    kw_write = dict(path=output_dir, zarr_flag=zarr)

    c.get_unused_data(raw_data=raw).write_file(
        filename="%(path)s/%(sign_type)s_untracked.nc", **kw_write
    )

    short_c = c._copy()
    short_c.shorter_than(size_max=nb_obs_min)
    c.longer_than(size_min=nb_obs_min)

    long_track = c.merge(raw_data=raw)
    short_track = short_c.merge(raw_data=raw)

    # We flag obs
    if c.virtual:
        long_track["virtual"][:] = long_track["time"] == 0
        long_track.normalize_longitude()
        long_track.filled_by_interpolation(long_track["virtual"] == 1)
        short_track["virtual"][:] = short_track["time"] == 0
        short_track.normalize_longitude()
        short_track.filled_by_interpolation(short_track["virtual"] == 1)

    logger.info("Longer track saved have %d obs", c.nb_obs_by_tracks.max())
    logger.info(
        "The mean length is %d observations for long track",
        c.nb_obs_by_tracks.mean(),
    )

    long_track.write_file(**kw_write)
    short_track.write_file(
        filename="%(path)s/%(sign_type)s_track_too_short.nc", **kw_write
    )


def get_group(
    dataset1,
    dataset2,
    index1,
    index2,
    score,
    invalid=2,
    low=10,
    high=60,
):
    group1, group2 = dict(), dict()
    m_valid = (score * 100) >= invalid
    i1, i2, score = index1[m_valid], index2[m_valid], score[m_valid] * 100
    # Eddies with no association & scores < invalid
    group1["nomatch"] = reverse_index(i1, len(dataset1))
    group2["nomatch"] = reverse_index(i2, len(dataset2))
    # Select all eddies involved in multiple associations
    i1_, nb1 = unique(i1, return_counts=True)
    i2_, nb2 = unique(i2, return_counts=True)
    i1_multi = i1_[nb1 >= 2]
    i2_multi = i2_[nb2 >= 2]
    m_multi = in1d(i1, i1_multi) + in1d(i2, i2_multi)
    group1["multi_match"] = unique(i1[m_multi])
    group2["multi_match"] = unique(i2[m_multi])

    # Low scores
    m_low = score <= low
    m_low *= ~m_multi
    group1["low"] = i1[m_low]
    group2["low"] = i2[m_low]
    # Intermediate scores
    m_i = (score > low) * (score <= high)
    m_i *= ~m_multi
    group1["intermediate"] = i1[m_i]
    group2["intermediate"] = i2[m_i]
    # High scores
    m_high = score > high
    m_high *= ~m_multi
    group1["high"] = i1[m_high]
    group2["high"] = i2[m_high]

    def get_twin(j2, j1):
        # True only if j1 is used only one
        m = bincount(j1)[j1] == 1
        # We keep only link of this mask j1 have exactly one parent
        j2_ = j2[m]
        # We count parent times
        m_ = (bincount(j2_)[j2_] == 2) * (bincount(j2)[j2_] == 2)
        # we fill first mask with second one
        m[m] = m_
        return m

    m1 = get_twin(i1, i2)
    m2 = get_twin(i2, i1)
    group1["parent"] = unique(i1[m1])
    group2["parent"] = unique(i2[m2])
    group1["twin"] = i1[m2]
    group2["twin"] = i2[m1]

    m = ~m1 * ~m2 * m_multi
    group1["complex"] = unique(i1[m])
    group2["complex"] = unique(i2[m])

    return group1, group2


def quick_compare():
    parser = EddyParser(
        "Tool to have a quick comparison between several identification"
    )
    parser.add_argument("ref", help="Identification file of reference")
    parser.add_argument("others", nargs="+", help="Identifications files to compare")
    parser.add_argument("--high", default=40, type=float)
    parser.add_argument("--low", default=20, type=float)
    parser.add_argument("--invalid", default=5, type=float)
    parser.contour_intern_arg()
    args = parser.parse_args()

    kw = dict(
        include_vars=[
            "longitude",
            *EddiesObservations.intern(args.intern, public_label=True),
        ]
    )

    ref = EddiesObservations.load_file(args.ref, **kw)
    print(f"[ref] {args.ref} -> {len(ref)} obs")
    groups_ref, groups_other = dict(), dict()
    others = {other: EddiesObservations.load_file(other, **kw) for other in args.others}
    for i, other_ in enumerate(args.others):
        other = others[other_]
        print(f"[{i}] {other_} -> {len(other)} obs")
        gr1, gr2 = get_group(
            ref,
            other,
            *ref.match(other, intern=args.intern),
            invalid=args.invalid,
            low=args.low,
            high=args.high,
        )
        groups_ref[other_] = gr1
        groups_other[other_] = gr2

    def display(value, ref=None):
        outs = list()
        for v in value:
            if ref:
                outs.append(f"{v/ref * 100:.1f}% ({v})")
            else:
                outs.append(v)
        return "".join([f"{v:^15}" for v in outs])

    keys = list(gr1.keys())
    print("     ", display(keys))
    for i, v in enumerate(groups_ref.values()):
        print(
            f"[{i:2}] ",
            display(
                (v_.sum() if v_.dtype == "bool" else v_.shape[0] for v_ in v.values()),
                ref=len(ref),
            ),
        )

    print(display(keys))
    for i, (k, v) in enumerate(groups_other.items()):
        print(
            f"[{i:2}] ",
            display(
                (v_.sum() if v_.dtype == "bool" else v_.shape[0] for v_ in v.values()),
                ref=len(others[k]),
            ),
        )
