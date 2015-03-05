# -*- coding: utf-8 -*-
# %run py_eddy_tracker_property_classes.py

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
import scipy.ndimage as nd
#import scipy.ndimage.morphology as morphology
#import scipy.ndimage.filters as filters
#import scipy.ndimage.label as label
import matplotlib.pyplot as plt




def pcol_2dxy(x, y):
    """
    Function to shift x, y for subsequent use with pcolor
    by Jeroen Molemaker UCLA 2008
    """
    mp, lp = x.shape
    m = mp - 1
    l = lp - 1
    x_pcol = np.zeros((mp, lp))
    y_pcol = np.zeros_like(x_pcol)
    x_tmp = 0.5 * (x[:, :l] + x[:, 1:lp])
    x_pcol[1:mp, 1:lp] = 0.5 * (x_tmp[:m] + x_tmp[1:mp])
    x_pcol[0] = 2. * (x_pcol[1] - x_pcol[2])
    x_pcol[:, 0] = 2. * (x_pcol[:, 1] - x_pcol[:, 2])
    y_tmp = 0.5 * (y[:, :l] + y[:, 1:lp])
    y_pcol[1:mp, 1:lp] = 0.5 * (y_tmp[:m] + y_tmp[1:mp])
    y_pcol[0] = 2. * (y_pcol[1] - y_pcol[2])
    y_pcol[:, 0] = 2. * (y_pcol[:, 1] - y_pcol[:, 2])
    return x_pcol, y_pcol


   
   
   
   
   
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
      
      grd:
        A grid object 
    """
    def __init__(self, contlon, contlat, eddy, grd):
        """
        """
        self.contlon = contlon.copy()
        self.contlat = contlat.copy()
        eddy.grd = grd # temporary fix
        self.eddy = eddy
        self.MLE = self.eddy.MAX_LOCAL_EXTREMA
        self.islice = slice(self.eddy.imin, self.eddy.imax)
        self.jslice = slice(self.eddy.jmin, self.eddy.jmax)
        self.sla = self.eddy.sla[self.jslice,
                                 self.islice].copy()
        
        if 'RectBivariate' in eddy.INTERP_METHOD:
            h0 = grd.sla_coeffs.ev(self.contlat[1:], self.contlon[1:])
        
        elif 'griddata' in eddy.INTERP_METHOD:
            
            #plt.figure()
            #plt.pcolormesh(grd.lon()[self.jslice, self.islice],
                           #grd.lat()[self.jslice, self.islice], self.sla)
            #plt.plot(self.contlon[1:], self.contlat[1:],'.-')
            #plt.show()
            
            points = np.array([grd.lon()[self.jslice, self.islice].ravel(),
                               grd.lat()[self.jslice, self.islice].ravel()]).T
            h0 = interpolate.griddata(points, self.sla.ravel(), (self.contlon[1:],
                                      self.contlat[1:]), 'linear')
        else:
            Exception
        
        self.h0_check = h0
        self.h0 = h0[np.isfinite(h0)].mean()
        self.amplitude = 0 #np.atleast_1d(0.)
        self.local_extrema = None #np.int(0)
        self.local_extrema_inds = None
        self.mask = self.eddy.mask_eff
        self.sla = np.ma.masked_where(self.mask == False, self.sla)
        self.num_features = np.atleast_1d(0)
    
    
    def within_amplitude_limits(self):
        """
        """
        return (self.amplitude >= self.eddy.AMPMIN and
                self.amplitude <= self.eddy.AMPMAX)
    
    
    def _set_cyc_amplitude(self):
        """
        """
        self.amplitude = self.h0
        self.amplitude -= self.sla.min()
        
    
    def _set_acyc_amplitude(self):
        """
        """
        self.amplitude = self.sla.max()
        self.amplitude -= self.h0
    
    
    def all_pixels_below_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are below a given SSH threshold for cyclonic eddies.
        """                  
        if np.any(self.sla > self.h0):
            return False # i.e., with self.amplitude == 0
        
        else:
            self._set_local_extrema(1)
            
            if (self.local_extrema > 0 and
                self.local_extrema <= self.MLE):
                self._set_cyc_amplitude()
            
            elif self.local_extrema > self.MLE:
                lmi_j, lmi_i = np.where(self.local_extrema_inds)
                levnm2 = level - (2 * self.eddy.INTERVAL)
                #print 'level, levnm2', level, levnm2
                slamin = np.atleast_1d(1e5)
                #print lmi_j, lmi_i
                for j, i in zip(lmi_j, lmi_i):
                    if slamin >= self.sla[j, i]:
                        slamin[:] = self.sla[j, i]
                        jmin, imin = j, i
                    if self.sla[j, i] >= levnm2:
                        self._set_cyc_amplitude()
                         # Prevent further calls to_set_acyc_amplitude
                        levnm2 = 1e5
                jmin += self.eddy.jmin
                imin += self.eddy.imin
                return (imin, jmin)
                
        return False
    
    
    def all_pixels_above_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """              
        if np.any(self.sla < self.h0):
            return False # i.e.,with self.amplitude == 0
        
        else:
            self._set_local_extrema(-1)
            
            if (self.local_extrema > 0 and
                self.local_extrema <= self.MLE):
                self._set_acyc_amplitude()
            
            elif self.local_extrema > self.MLE:
                lmi_j, lmi_i = np.where(self.local_extrema_inds)
                levnp2 = level + (2 * self.eddy.INTERVAL)
                slamax = np.atleast_1d(-1e5)
                for j, i in zip(lmi_j, lmi_i):
                    if slamax <= self.sla[j, i]:
                        slamax[:] = self.sla[j, i]
                        jmax, imax = j, i
                    if self.sla[j, i] <= levnp2:
                        self._set_acyc_amplitude()
                        levnp2 = -1e5
                jmax += self.eddy.jmin
                imax += self.eddy.imin
                return (imax, jmax)
                
        return False
    
    
    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        #local_extrema = np.ma.masked_where(self.mask == False, self.sla)
        local_extrema = self.sla.copy()
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
        neighborhood = nd.morphology.generate_binary_structure(arr.ndim, 2)
        # Get local mimima
        detected_minima = (nd.filters.minimum_filter(arr,
                           footprint=neighborhood) == arr)
        background = (arr == 0)
        eroded_background = nd.morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima -= eroded_background
        self.local_extrema_inds = detected_minima
        self.local_extrema = detected_minima.sum()
        return self
        
        
    def debug_figure(self, grd):
        """
        Uncomment in py-eddy-tracker-classes.py
        """
        if self.local_extrema >= 1 and self.amplitude:
            
            plt.figure(587)
            cmin, cmax = -8, 8
            ##cmin, cmax = (self.h0_check.min() - 2,
                          ##self.h0_check.max() + 2)
            #cmin, cmax = (self.sla.min(), self.sla.max())
            cm = plt.cm.gist_ncar
            #cm = plt.cm.hsv
            
            plt.title('Local max/min count: %s, Amp: %s' % (
                             self.local_extrema, self.amplitude))
            
            x, y = (grd.lon()[self.jslice, self.islice],
                    grd.lat()[self.jslice, self.islice])
            #x, y = pcol_2dxy(x, y)
            pcm = plt.pcolormesh(x, y, self.sla.data, cmap=cm)
            plt.clim(cmin, cmax)
            plt.plot(self.contlon, self.contlat)
            plt.scatter(self.contlon[1:], self.contlat[1:], s=100, c=self.h0_check,
                        cmap=cm, vmin=cmin, vmax=cmax)
            #plt.scatter(centlon_lmi, centlat_lmi, c='k')
            #plt.scatter(centlon_e, centlat_e, c='w')
            print self.local_extrema
            print self.local_extrema_inds
            lmi_j, lmi_i = np.where(self.local_extrema_inds)
            #lmi_i = lmi_i[0] + self.eddy.imin
            #lmi_j = lmi_j[0] + self.eddy.jmin
            #print lmi_i
            lmi_i = np.array(lmi_i) + self.eddy.imin
            lmi_j = np.array(lmi_j) + self.eddy.jmin
            x_i, y_i = grd.lon()[lmi_j, lmi_i], grd.lat()[lmi_j, lmi_i]
            plt.scatter(x_i, y_i, s=40, c='w')
            plt.axis('image')
            plt.colorbar(pcm)
            plt.show()
        return

        
        
class SwirlSpeed(object):
    """
    Class to calculate average geostrophic velocity along
    a contour, *uavg*, and return index to contour with maximum
    *uavg* within a series of closed contours.
    
    Attributes:
      contour:
        A matplotlib contour object of high-pass filtered SLA
        
      eddy:
        A tracklist object holding the SLA data
      
      grd:
        A grid object 
    """
    def __init__(self, contour):
        """
        ci : index to contours
        li : index to levels
        """
        x, y, ci, li = [], [], [], []

        for lind, cont in enumerate(contour.collections):
            for cind, coll in enumerate(cont.get_paths()):
                x.append(coll.vertices[:, 0])
                y.append(coll.vertices[:, 1])
                ci.append(len(coll.vertices[:, 0]))
            li.append(len(cont.get_paths()))
    
        self.x = np.array([val for sublist in x for val in sublist])
        self.y = np.array([val for sublist in y for val in sublist])
        self.nb_pt_per_c = np.array(ci)
        self.ci = self.nb_pt_per_c.cumsum() - self.nb_pt_per_c
        self.nb_c_per_l = np.array(li)
        self.li = self.nb_c_per_l.cumsum() - self.nb_c_per_l
        self.level_slice = None
        self.nearesti = None # index to nearest contour
        
    
    def _set_level_slice(self, thelevel):
        """
        Set slices
        """
        if self.nb_c_per_l[thelevel] == 0:
            self.level_slice = None
        else:
            self.level_view_of_contour = slice(self.li[thelevel],
                self.li[thelevel] + self.nb_c_per_l[thelevel])
            nb_pts = self.nb_pt_per_c[self.level_view_of_contour].sum()
            self.level_slice = slice(self.ci[self.li[thelevel]],
                self.ci[self.li[thelevel]] + nb_pts)
        return self
        
    
    def set_dist_array_size(self, thelevel):
        """
        """
        self._set_level_slice(thelevel)
        return self
        
    
    def set_nearest_contour_index(self, xpt, ypt):
        """
        """
        self.dist = (self.x[self.level_slice] - xpt)**2
        self.dist += (self.y[self.level_slice] - ypt)**2
        try:
            self.nearesti = self.dist.argmin()
        except:
            self.nearesti = None
        return self
    
    
    def get_index_nearest_path(self):
        """
        """
        if self.nearesti is not None:
            indices_of_first_pts = self.ci[self.level_view_of_contour]
            for i, index_of_first_pt in enumerate(indices_of_first_pts):
                if ((index_of_first_pt - 
                     indices_of_first_pts[0]) > self.nearesti):
                    return i - 1
            return i
        else:
            return False
    
    