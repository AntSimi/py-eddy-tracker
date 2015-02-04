# -*- coding: utf-8 -*-
# %run py_eddy_tracker_amplitude.py

"""
===========================================================================
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

Copyright (c) 2015 by Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================


py_eddy_tracker_amplitude.py

Version 1.4.2


===========================================================================

"""
from scipy import interpolate
import numpy as np
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters


class EddyProperty (object):
    """
    Class to hold eddy properties *amplitude* and counts of *local maxima/minima*
    within a closed region of a sea level anomaly field.
    
    Variables:
      centlon:
        Longitude centroid coordinate
      
      centlat:
        Latitude centroid coordinate
      
      eddy_radius_s:
        Speed based radius
      
      eddy_radius_e:
        Effective radius
      
      amplitude:
        Eddy amplitude
      
      uavg:
        Average eddy swirl speed
      
      teke:
        Average eddy kinetic energy within eddy
      
      rtime:
        Time
    """
    def __init__(self):
        """
        """
        self.centlon = 0
        self.centlat = 0
        self.eddy_radius_s = 0
        self.eddy_radius_e = 0
        self.amplitude = 0
        self.uavg = 0
        self.teke = 0
        self.rtime = 0

        

class Amplitude (object):
    """
    Class to calculate *amplitude* and counts of *local maxima/minima*
    within a closed region of a sea level anomaly field.
    
    Attributes:
      contlon:
        Longitude coordinates of contour
      
      contlat:
        Latitude coordinates of contour
      
      eddy:
        A tracklist object holding the SLA data
      
      mask
        A 2d mask identifying the 
    """
    def __init__(self, contlon, contlat, eddy, grd):
        """
        """
        self.contlon = contlon.copy()
        self.contlat = contlat.copy()
        eddy.grd = grd # temporary fix
        self.eddy = eddy
        self.MLE = self.eddy.MAX_LOCAL_EXTREMA
        self.imin, self.imax = self.eddy.imin, self.eddy.imax
        self.jmin, self.jmax = self.eddy.jmin, self.eddy.jmax
        self.sla = self.eddy.sla[self.jmin:self.jmax,
                                 self.imin:self.imax].copy()
        self.rbspline = grd.sla_coeffs
        h0 = self.rbspline.ev(self.contlat, self.contlon)
        self.h0 = h0[np.isfinite(h0)].mean()
        self.amplitude = np.atleast_1d(0.)
        self.local_extrema = np.int(0)
        #print 'self.sla', self.sla.shape
        #print 'self.eddy.mask_eff_1d', self.eddy.mask_eff_1d.shape
        self.mask = self.eddy.mask_eff
        #print 'self.mask', self.mask.shape
        #self.sla = self.sla[self.mask]
        self.sla = np.ma.masked_where(self.mask == False, self.sla)
    
    
    def within_amplitude_limits(self):
        """
        """
        return (self.amplitude >= self.eddy.AMPMIN and
                self.amplitude <= self.eddy.AMPMAX)
    
    
    def all_pixels_below_h0(self):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are below a given SSH threshold for cyclonic eddies.
        """                  
        if np.any(self.sla > self.h0):
            return self # amplitude == 0
        else:
            self._set_local_extrema(1)
            if (self.local_extrema > 0 and
                self.local_extrema <= self.MLE):
                self.amplitude = self.h0
                self.amplitude -= self.sla.min()
        return self
    
    
    def all_pixels_above_h0(self):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """              
        if np.any(self.sla < self.h0):
            #print '--- h0 %s; slamax %s' % (self.h0, self.sla.max())
            return self # amplitude == 0
        else:
            self._set_local_extrema(-1)
            if (self.local_extrema > 0 and
                self.local_extrema <= self.MLE):
                #print '--- h0 %s; slamax %s' % (self.h0, self.sla.max())
                self.amplitude = self.sla.max()
                self.amplitude -= self.h0
        return self
    
    
    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        mask = self.mask
        local_extrema = np.ma.masked_where(mask == False, self.sla)
        local_extrema *= sign
        self._detect_local_minima(local_extrema)
        return self
    
    
    def _detect_local_minima(self, arr):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
        # Get local mimima
        detected_minima = (filters.minimum_filter(arr,
                           footprint=neighborhood) == arr)
        background = (arr == 0)
        eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima -= eroded_background
        # NOTE: by summing we lose the indices (they could be useful)
        self.local_extrema = detected_minima.sum()
        return self
        
        
    def debug_figure(self):
        """
        """
        plt.figure()
        if 'Cyclonic' in self.eddy.sign_type:
            plt.title('Cyclones')
            sla = self.eddy.slacopy - self.eddy.slacopy.mean()
        
        x, y = self.grd.lon(), self.grd.lat()
        plt.pcolormesh(x, y, sla)
        plt.clim(-10, 10)
        plt.scatter(centlon_lmi, centlat_lmi, c='k')
        plt.scatter(centlon_e, centlat_e, c='w')
        lmi_j, lmi_i = np.where(self.local_extrema)
        lmi_i = lmi_i[0] + self.grd.imin
        lmi_j = lmi_j[0] + self.grd.jmin
        x_i, y_i = self.grd.lon()[lmi_j, lmi_i], self.grd.lat()[lmi_j, lmi_i]
        plt.scatter(x_i, y_i, c='gray')
        plt.show()

        
        
class SwirlSpeed(object):
    """
    """
    def __init__(self, contour):
        """
        ci : index to contours
        li : index to levels
        """
        self.x = []
        self.y = []
        self.ci = []
        self.li = []

        for lind, cont in enumerate(contour.collections):
            for cind, coll in enumerate(cont.get_paths()):
                self.x.append(coll.vertices[:, 0])
                self.y.append(coll.vertices[:, 1])
                thelen = len(coll.vertices[:, 0])
                self.ci.append([cind] * thelen)
                self.li.append([lind] * thelen)
    
        self.x = np.array([val for sublist in self.x for val in sublist])
        self.y = np.array([val for sublist in self.y for val in sublist])
        self.ci = np.array([val for sublist in self.ci for val in sublist],
                                                               dtype=np.int)
        self.li = np.array([val for sublist in self.li for val in sublist],
                                                               dtype=np.int)
        
        self.nearesti = None
        self.dist = None
        self.lbool = None
        
    
    def _set_level_bool(self, thelevel):
        """
        Set a boolean to select only values to a discrete
        contour level in self.li
        """
        self.lbool = self.li == thelevel
        return self
        
    
    def set_dist_array_size(self, thelevel):
        """
        """
        self._set_level_bool(thelevel)
        self.dist = np.empty_like(self.li[self.lbool])
        return self
        
    
    def set_nearest_contour_index(self, xpt, ypt):
        """
        """
        self.dist[:] = (self.x[self.lbool] - xpt)**2
        self.dist += (self.y[self.lbool] - ypt)**2
        try:
            self.nearesti = self.dist.argmin()
        except:
            self.nearesti = None
        return self
    
    
    def get_index_nearest_path(self):
        """
        """
        ind = self.nearesti
        if ind is not None:
            return self.ci[self.lbool[ind]]
        else:
            return False
    
    
        
        
        