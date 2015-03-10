# -*- coding: utf-8 -*-
# %run make_eddy_track_CLS.py

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
#import pickle
import cPickle as pickle
#import dill
import numpy as np
from make_eddy_track_AVISO import track_eddies, AvisoGrid




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    DATA_DIR = '/data/OCE_ETU/MSA/emason/Global_DT10/'
    
    A_PKL_FILES = 'A_eddy_????????.pkl'
    C_PKL_FILES = 'C_eddy_????????.pkl'
    
    A_PKL_FILES = sorted(glob.glob(DATA_DIR + A_PKL_FILES))
    C_PKL_FILES = sorted(glob.glob(DATA_DIR + C_PKL_FILES))
    
    
    for PKL_FILES in (A_PKL_FILES, C_PKL_FILES):
        
        for active, PKL_FILE in enumerate(PKL_FILES):
            
            print PKL_FILE
            
            # Unpickle
            with open(PKL_FILE, 'rb') as the_pickle:
                eddy = pickle.load(the_pickle)
                print '--- loaded %s' % PKL_FILE
            
            #print 'eddy.index', eddy.index
            eddy.savedir = DATA_DIR + eddy.savedir.rpartition('/')[-1]

            eddy.TRACK_DURATION_MIN = 10
            
            
            
            
            if active:
                
                first_record = False
                
                eddy.new_list = False
                eddy.tracklist = tracklist.tolist()
                
                eddy.index = index
                eddy.ch_index = ch_index
                eddy.ncind = ncind
                
                eddy.old_lon = old_lon
                eddy.old_lat = old_lat
                eddy.old_amp = old_amp
                eddy.old_uavg = old_uavg
                eddy.old_radii_s = old_radii_s
                eddy.old_radii_e = old_radii_e
                eddy.old_teke = old_teke
                #eddy.old_temp = old_temp
                #eddy.old_salt = old_salt
                
                eddy = track_eddies(eddy, first_record)
                tracklist = np.copy(eddy.tracklist)
                
            else:
                
                first_record = True
                
                eddy.create_netcdf(DATA_DIR, eddy.savedir)
                eddy.set_old_variables()
                
                eddy = track_eddies(eddy, first_record)
                tracklist = np.copy(eddy.tracklist)
            
            
            print 'eddy.index', eddy.index
            print len(eddy.tracklist)
            
            
            if not first_record:
                print eddy.new_time_tmp[0]
                eddy.write2netcdf(eddy.new_time_tmp[0])
            
            index = eddy.index
            ch_index = eddy.ch_index
            ncind = eddy.ncind
            
            old_lon = eddy.old_lon
            old_lat = eddy.old_lat
            old_amp = eddy.old_amp
            old_uavg = eddy.old_uavg
            old_radii_s = eddy.old_radii_s
            old_radii_e = eddy.old_radii_e
            old_teke = eddy.old_teke
            #old_temp = eddy.old_temp
            #old_salt = eddy.old_salt
            
            
            #eddy.reset_holding_variables()
            
            
            
            