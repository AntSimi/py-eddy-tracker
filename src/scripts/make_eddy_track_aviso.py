# -*- coding: utf-8 -*-

"""
===============================================================================
This file is part of py-eddy-tracker.

    py-eddy-tracker is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    py-eddy-tracker is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with py-eddy-tracker.  If not, see <http://www.gnu.org/licenses/>.

Copyright (c) 2014-2015 by Evan Mason
Email: emason@imedea.uib-csic.es
===============================================================================

make_eddy_track_AVISO.py

Version 2.0.3

===============================================================================
"""

from sys import argv
from glob import glob
from yaml import load as yaml_load
from datetime import datetime
from netCDF4 import Dataset
from matplotlib import use as mpl_use
mpl_use('Agg')
from matplotlib.dates import date2num, num2date
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cPickle as pickle
import logging
import numpy as np

from py_eddy_tracker.py_eddy_tracker_classes import track_eddies, okubo_weiss, \
    gaussian_resolution, collection_loop, func_hann2d_fast, datestr2datetime
from py_eddy_tracker.py_eddy_tracker_property_classes import SwirlSpeed
from py_eddy_tracker.aviso_grid import AvisoGrid
from py_eddy_tracker.make_eddy_tracker_list_obj import SearchEllipse, TrackList
from py_eddy_tracker.global_tracking import GlobalTracking


class ColoredFormatter(logging.Formatter):
    COLOR_LEVEL = dict(
        CRITICAL="\037[37;41m",
        ERROR="\033[31;47m",
        WARNING="\033[30;47m",
        MAIN_INFO="\033[36m",
        INFO="\033[36m",
        DEBUG="\033[34m",
        )

    def __init__(self, message):
        super(ColoredFormatter, self).__init__(message)

    def format(self, record):
        color = self.COLOR_LEVEL.get(record.levelname, '')
        color_reset = '\033[0m'
        model = color + '%s' + color_reset
        record.msg = model % record.msg
        record.funcName = model % record.funcName
        record.module = model % record.module
        record.levelname = model % record.levelname
        return super(ColoredFormatter, self).format(record)

if __name__ == '__main__':
    FORMAT_LOG = "%(levelname)-8s %(asctime)s %(module)s." \
                 "%(funcName)s :\n\t\t\t\t\t%(message)s"
    # set up logging to CONSOLE
    CONSOLE = logging.StreamHandler()
    CONSOLE.setFormatter(ColoredFormatter(FORMAT_LOG))
    # add the handler to the root logger
    logging.getLogger('').addHandler(CONSOLE)

    # Run using:
    try:
        YAML_FILE = argv[1]
    except Exception:
        logging.error("To run use 'make_eddy_track_AVISO.py "
                      "eddy_tracker_configuration.yaml'")
        exit(1)

    # Read yaml configuration file
    with open(YAML_FILE, 'r') as stream:
        CONFIG = yaml_load(stream)

    logging.getLogger().setLevel(getattr(logging, CONFIG['VERBOSE'].upper()))
    VERBOSE = 'No' if CONFIG['VERBOSE'].upper() != 'DEBUG' else 'Yes'

    logging.info('Launching with yaml file: %s', YAML_FILE)

    # Setup configuration
    DATA_DIR = CONFIG['PATHS']['DATA_DIR']
    SAVE_DIR = CONFIG['PATHS']['SAVE_DIR']
    logging.info('Outputs saved to %s', SAVE_DIR)
    RW_PATH = CONFIG['PATHS']['RW_PATH']

    DIAGNOSTIC_TYPE = CONFIG['DIAGNOSTIC_TYPE']

    CONFIG['THE_DOMAIN'] = CONFIG['DOMAIN']['THE_DOMAIN']

    # It is not recommended to change values given below
    # for 'Global', 'BlackSea' or 'MedSea'...
    if 'Global' in CONFIG['THE_DOMAIN']:
        CONFIG['lonmin'] = -100.
        CONFIG['lonmax'] = 290.
        CONFIG['latmin'] = -80.
        CONFIG['latmax'] = 80.

    elif CONFIG['THE_DOMAIN'] in ('Regional', 'MedSea'):
        CONFIG['lonmin'] = CONFIG['DOMAIN']['LONMIN']
        CONFIG['lonmax'] = CONFIG['DOMAIN']['LONMAX']
        CONFIG['latmin'] = CONFIG['DOMAIN']['LATMIN']
        CONFIG['latmax'] = CONFIG['DOMAIN']['LATMAX']

    DATE_STR = CONFIG['DATE_STR'] = CONFIG['DOMAIN']['DATE_STR']
    DATE_END = CONFIG['DATE_END'] = CONFIG['DOMAIN']['DATE_END']

    AVISO_DT14 = CONFIG['AVISO']['AVISO_DT14']
    if AVISO_DT14:
        PRODUCT = 'AVISO_DT14'
        AVISO_DT14_SUBSAMP = CONFIG['AVISO']['AVISO_DT14_SUBSAMP']
        if AVISO_DT14_SUBSAMP:
            DAYS_BTWN_RECORDS = CONFIG['AVISO']['DAYS_BTWN_RECORDS']
        else:
            DAYS_BTWN_RECORDS = 1.
    else:
        PRODUCT = 'AVISO_DT10'
        DAYS_BTWN_RECORDS = 7.  # old seven day AVISO
    CONFIG['DAYS_BTWN_RECORDS'] = DAYS_BTWN_RECORDS
    AVISO_FILES = CONFIG['AVISO']['AVISO_FILES']

    # TRACK_DURATION_MIN = CONFIG['TRACK_DURATION_MIN']

    if 'SLA' in DIAGNOSTIC_TYPE:
        MAX_SLA = CONFIG['CONTOUR_PARAMETER'
                         ]['CONTOUR_PARAMETER_SLA']['MAX_SLA']
        INTERVAL = CONFIG['CONTOUR_PARAMETER'
                          ]['CONTOUR_PARAMETER_SLA']['INTERVAL']
        CONFIG['CONTOUR_PARAMETER'] = np.arange(-MAX_SLA, MAX_SLA + INTERVAL,
                                                INTERVAL)
        CONFIG['SHAPE_ERROR'] = np.full(CONFIG['CONTOUR_PARAMETER'].size,
                                        CONFIG['SHAPE_ERROR'])

    elif 'Q' in DIAGNOSTIC_TYPE:
        MAX_Q = CONFIG['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_Q']['MAX_Q']
        NUM_LEVS = CONFIG['CONTOUR_PARAMETER'
                          ]['CONTOUR_PARAMETER_Q']['NUM_LEVS']
        CONFIG['CONTOUR_PARAMETER'] = np.linspace(0, MAX_Q, NUM_LEVS)[::-1]
    else:
        Exception

    # JDAY_REFERENCE = CONFIG['JDAY_REFERENCE']

    # RADMIN = CONFIG['RADMIN']
    # RADMAX = CONFIG['RADMAX']

    if 'SLA' in DIAGNOSTIC_TYPE:
        # AMPMIN = CONFIG['AMPMIN']
        # AMPMAX = CONFIG['AMPMAX']
        pass
    elif 'Q' in DIAGNOSTIC_TYPE:
        AMPMIN = 0.02  # max(abs(xi/f)) within the eddy
        AMPMAX = 100.
    else:
        Exception

    SAVE_FIGURES = CONFIG['SAVE_FIGURES']

    SMOOTHING = CONFIG['SMOOTHING']
    if SMOOTHING:
        if 'SLA' in DIAGNOSTIC_TYPE:
            ZWL = np.atleast_1d(CONFIG['SMOOTHING_SLA']['ZWL'])
            MWL = np.atleast_1d(CONFIG['SMOOTHING_SLA']['MWL'])
        elif 'Q' in DIAGNOSTIC_TYPE:
            SMOOTH_FAC = CONFIG['SMOOTHING_Q']['SMOOTH_FAC']
        else:
            Exception
        SMOOTHING_TYPE = CONFIG['SMOOTHING_SLA']['TYPE']

    if 'Q' in DIAGNOSTIC_TYPE:
        AMP0 = 0.02  # vort/f
    elif 'SLA' in DIAGNOSTIC_TYPE:
        AMP0 = CONFIG['AMP0']
    TEMP0 = CONFIG['TEMP0']
    SALT0 = CONFIG['SALT0']

    # EVOLVE_AMP_MIN = CONFIG['EVOLVE_AMP_MIN']
    # EVOLVE_AMP_MAX = CONFIG['EVOLVE_AMP_MAX']
    # EVOLVE_AREA_MIN = CONFIG['EVOLVE_AREA_MIN']
    # EVOLVE_AREA_MAX = CONFIG['EVOLVE_AREA_MAX']

    # End user configuration setup options
    # --------------------------------------------------------------------------

    try:
        assert DATE_STR < DATE_END, 'DATE_END must be larger than DATE_STR'
        assert DIAGNOSTIC_TYPE in ('Q', 'SLA'), 'Undefined DIAGNOSTIC_TYPE'
    except AssertionError, msg:
        logging.error(msg)
        raise
    START_DATE = date2num(datestr2datetime(str(DATE_STR)))
    END_DATE = date2num(datestr2datetime(str(DATE_END)))

    # Get complete AVISO file list
    AVISO_FILES = sorted(glob(DATA_DIR + AVISO_FILES))

    # For subsampling to get identical list as old_AVISO use:
    # AVISO_FILES = AVISO_FILES[5:-5:7]
    if AVISO_DT14 and AVISO_DT14_SUBSAMP:
        AVISO_FILES = AVISO_FILES[5:-5:np.int(DAYS_BTWN_RECORDS)]

    # Set up a grid object using first AVISO file in the list
    SLA_GRD = AvisoGrid(AVISO_FILES[0], CONFIG['THE_DOMAIN'], PRODUCT,
                        CONFIG['lonmin'], CONFIG['lonmax'],
                        CONFIG['latmin'], CONFIG['latmax'])

    # Instantiate search ellipse object
    SEARCH_ELLIPSE = SearchEllipse(CONFIG['THE_DOMAIN'],
                                   SLA_GRD, DAYS_BTWN_RECORDS,
                                   RW_PATH)

    if 'Gaussian' in SMOOTHING_TYPE:
        # Get parameters for gaussian_filter
        ZRES, MRES = gaussian_resolution(SLA_GRD.resolution,
                                         ZWL, MWL)

    plt.figure(1)

    if 'Q' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = SAVE_DIR + 'eddy_tracks_Q_AVISO_anticyclonic.nc'
        C_SAVEFILE = SAVE_DIR + 'eddy_tracks_Q_AVISO_cyclonic.nc'

    elif 'SLA' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = SAVE_DIR + 'eddy_tracks_SLA_AVISO_anticyclonic.nc'
        C_SAVEFILE = SAVE_DIR + 'eddy_tracks_SLA_AVISO_cyclonic.nc'

    # Initialise two eddy objects to hold data
    # kwargs = CONFIG
    A_EDDY = TrackList('Anticyclonic', A_SAVEFILE,
                       SLA_GRD, SEARCH_ELLIPSE, **CONFIG)

    C_EDDY = TrackList('Cyclonic', C_SAVEFILE,
                       SLA_GRD, SEARCH_ELLIPSE, **CONFIG)

    A_EDDY.search_ellipse = SEARCH_ELLIPSE
    C_EDDY.search_ellipse = SEARCH_ELLIPSE

    # See Chelton section B2 (0.4 degree radius)
    # These should give 8 and 1000 for 0.25 deg resolution
    PIXMIN = np.round((np.pi * CONFIG['RADMIN'] ** 2) /
                      SLA_GRD.resolution ** 2)
    PIXMAX = np.round((np.pi * CONFIG['RADMAX'] ** 2) /
                      SLA_GRD.resolution ** 2)
    logging.info('Pixel range = %s-%s', np.int(PIXMIN), np.int(PIXMAX))

    A_EDDY.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
    C_EDDY.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]

    # Create nc files for saving of eddy tracks
    A_EDDY.create_netcdf(DATA_DIR, A_SAVEFILE)
    C_EDDY.create_netcdf(DATA_DIR, C_SAVEFILE)

    # Loop through the AVISO files...
    START = True

    START_TIME = datetime.now()

    logging.info('Start tracking')

    NB_STEP = 0
    for AVISO_FILE in AVISO_FILES:
        with Dataset(AVISO_FILE) as nc:

            try:
                date = nc.OriginalName
                if 'qd_' in date:
                    date = date.partition('qd_')[2].partition('_')[0]
                else:
                    date = date.partition('h_')[2].partition('_')[0]
                date = datestr2datetime(date)
                date = date2num(date)
            except Exception:
                date = nc.variables['time'][:]
                date += SLA_GRD.base_date

        rtime = date

        if date < START_DATE or date > END_DATE:
            continue

        logging.info('AVISO_FILE : %s', AVISO_FILE)

        # Holding variables
        A_EDDY.reset_holding_variables()
        C_EDDY.reset_holding_variables()

        sla = SLA_GRD.get_aviso_data(AVISO_FILE)
        SLA_GRD.set_mask(sla).uvmask()

        if SMOOTHING:
            if 'Gaussian' in SMOOTHING_TYPE:
                logging.info('applying Gaussian high-pass filter')
                # Set landpoints to zero
                np.place(sla, SLA_GRD.mask == 0, 0.)
                if hasattr(sla, 'data'):
                    np.place(sla, sla.data == SLA_GRD.fillval, 0.)
                # High pass filter, see
                # http://stackoverflow.com/questions/6094957/high-pass-filter-
                # for-image-processing-in-python-by-using-scipy-numpy
                sla -= gaussian_filter(sla, [MRES, ZRES])

            elif 'Hanning' in SMOOTHING_TYPE:
                logging.info('applying %s passes of Hanning filter',
                             SMOOTH_FAC)
                # Do SMOOTH_FAC passes of 2d Hanning filter
                sla = func_hann2d_fast(sla, SMOOTH_FAC)

            else:
                raise Exception('Filter unknown : %s', SMOOTHING_TYPE)

        # Apply the landmask
        sla = np.ma.masked_where(SLA_GRD.mask == 0, sla)

        # Get timing
        date = num2date(rtime)
        yr = date.year
        mo = date.month
        da = date.day

        # Multiply by 0.01 for m
        SLA_GRD.set_geostrophic_velocity(sla * 0.01)

        # Remove padded boundary
        sla = sla[SLA_GRD.jup0:SLA_GRD.jup1, SLA_GRD.iup0:SLA_GRD.iup1]

        # Calculate EKE
        SLA_GRD.getEKE()

        if 'SLA' in DIAGNOSTIC_TYPE:

            A_EDDY.sla = sla.copy()
            C_EDDY.sla = sla.copy()
            A_EDDY.slacopy = sla.copy()
            C_EDDY.slacopy = sla.copy()

        # Get scalar speed
        uspd = np.sqrt(SLA_GRD.u_val ** 2 + SLA_GRD.v_val ** 2)
        uspd = np.ma.masked_where(
            SLA_GRD.mask[SLA_GRD.jup0:SLA_GRD.jup1,
                         SLA_GRD.iup0:SLA_GRD.iup1] == 0,
            uspd)
        A_EDDY.uspd = uspd.copy()
        C_EDDY.uspd = uspd.copy()

        # Set interpolation coefficients
        SLA_GRD.set_interp_coeffs(sla, uspd)
        A_EDDY.sla_coeffs = SLA_GRD.sla_coeffs
        A_EDDY.uspd_coeffs = SLA_GRD.uspd_coeffs
        C_EDDY.sla_coeffs = SLA_GRD.sla_coeffs
        C_EDDY.uspd_coeffs = SLA_GRD.uspd_coeffs

        # Get contours of Q/sla parameter
        if 'first_record' not in locals():
            contfig = plt.figure(99)
            ax = contfig.add_subplot(111)

        if 'SLA' in DIAGNOSTIC_TYPE:
            logging.info('Processing SLA contours for eddies')
            A_CS = ax.contour(SLA_GRD.lon,
                              SLA_GRD.lat,
                              A_EDDY.sla, A_EDDY.CONTOUR_PARAMETER)
            # Note that C_CS is in reverse order
            C_CS = ax.contour(SLA_GRD.lon,
                              SLA_GRD.lat,
                              C_EDDY.sla, C_EDDY.CONTOUR_PARAMETER)

        else:
            raise Exception()

        # clear the current axis
        ax.cla()

        # Set contour coordinates and indices for calculation of
        # speed-based radius
        A_EDDY.swirl = SwirlSpeed(A_CS)
        C_EDDY.swirl = SwirlSpeed(C_CS)

        # Now we loop over the CS collection
        if 'SLA' in DIAGNOSTIC_TYPE:
            A_EDDY = collection_loop(A_CS, SLA_GRD, rtime,
                                     a_list_obj=A_EDDY, c_list_obj=None,
                                     sign_type=A_EDDY.SIGN_TYPE)
            # Note that C_CS is reverse order
            C_EDDY = collection_loop(C_CS, SLA_GRD, rtime,
                                     a_list_obj=None, c_list_obj=C_EDDY,
                                     sign_type=C_EDDY.SIGN_TYPE)

        ymd_str = ''.join((str(yr), str(mo).zfill(2), str(da).zfill(2)))

        # Test pickling

        save2nc = GlobalTracking(A_EDDY, ymd_str)
        save2nc.write_netcdf()
        save2nc = GlobalTracking(C_EDDY, ymd_str)
        save2nc.write_netcdf()
        NB_STEP += 1

        if START:
            first_record = True
            # Set old variables equal to new variables
            A_EDDY.set_old_variables()
            C_EDDY.set_old_variables()
            START = False
            logging.info('Tracking eddies')
        else:
            first_record = False

        # Track the eddies
        A_EDDY = track_eddies(A_EDDY, first_record)
        C_EDDY = track_eddies(C_EDDY, first_record)

        # Save inactive eddies to nc file
        if not first_record:

            logging.debug('saving to nc %s', A_EDDY.SAVE_DIR)
            logging.debug('saving to nc %s', C_EDDY.SAVE_DIR)
            logging.debug('+++')

            A_EDDY.write2netcdf(rtime)
            C_EDDY.write2netcdf(rtime)

    A_EDDY.kill_all_tracks()
    C_EDDY.kill_all_tracks()

    A_EDDY.write2netcdf(rtime, stopper=1)
    C_EDDY.write2netcdf(rtime, stopper=1)

    # Total running time
    logging.info('Duration by loop : %s',
                 (datetime.now() - START_TIME) / NB_STEP)
    logging.info('Duration : %s', datetime.now() - START_TIME)

    logging.info('Outputs saved to %s', SAVE_DIR)
