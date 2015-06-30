#!/usr/bin/env python
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

Copyright (c) 2014 by Evan Mason
Email: emason@imedea.uib-csic.es
===============================================================================

make_eddy_track_CLS.py

Version 1.4.2


===============================================================================
"""
import glob as glob
import cPickle as pickle
import numpy as np
from make_eddy_track_AVISO import track_eddies
import logging

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    TRACK_DURATION_MIN = 28

    # DATA_DIR = '/data/OCE_ETU/MSA/emason/Global_DT10/'
    # DATA_DIR = '/Users/emason/mercurial_projects/py-eddy-tracker-cls/py-eddy-tracker-cls/outputs/'
    DATA_DIR = '/home/emason/Downloads/'

    # A_PKL_FILES = 'A_eddy_????????.pkl'
    C_PKL_FILES = 'C_eddy_????????.pkl'

    # PKL_FILES = glob.glob(DATA_DIR + A_PKL_FILES)
    PKL_FILES = glob.glob(DATA_DIR + C_PKL_FILES)

    PKL_FILES.sort()
    for active, PKL_FILE in enumerate(PKL_FILES):

        try:
            del eddy
        except:
            pass

        # Unpickle
        with open(PKL_FILE, 'rb') as the_pickle:
            eddy = pickle.load(the_pickle)
            logging.info('\tloaded %s', PKL_FILE)

        eddy.savedir = DATA_DIR + eddy.savedir.rpartition('/')[-1]

        eddy.TRACK_DURATION_MIN = TRACK_DURATION_MIN

        if active:

            eddy.new_list = False
            eddy.tracklist = tracklist  # .tolist()

            try:
                eddy.index = index
                eddy.ch_index = ch_index
                # eddy.ncind = ncind
            except:
                pass
            eddy.old_lon = old_lon
            eddy.old_lat = old_lat
            eddy.old_amp = old_amp
            eddy.old_uavg = old_uavg
            eddy.old_radii_s = old_radii_s
            eddy.old_radii_e = old_radii_e
            eddy.old_teke = old_teke

            eddy = track_eddies(eddy, first_record)

            try:
                eddy.ncind = ncind
            except:
                pass

        else:
            eddy.create_netcdf(DATA_DIR, eddy.savedir)
            eddy.set_old_variables()
            first_record = True
            eddy = track_eddies(eddy, first_record)
            first_record = False
            tracklist = eddy.tracklist

        if not first_record:
            eddy.write2netcdf(eddy.new_time_tmp[0])
            # tracklist is modified by write2netcdf, so
            # place update just after
            tracklist = eddy.tracklist
            ncind = np.copy(eddy.ncind)
            ch_index = np.copy(eddy.ch_index)
            index = np.copy(eddy.index)

        old_lon = eddy.old_lon
        old_lat = eddy.old_lat
        old_amp = eddy.old_amp
        old_uavg = eddy.old_uavg
        old_radii_s = eddy.old_radii_s
        old_radii_e = eddy.old_radii_e
        old_teke = eddy.old_teke
        # old_temp = eddy.old_temp
        # old_salt = eddy.old_salt

        eddy.reset_holding_variables()
