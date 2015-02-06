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
        x = []
        y = []
        ci = []
        li = []

        for lind, cont in enumerate(contour.collections):
            for cind, coll in enumerate(cont.get_paths()):
                x.append(coll.vertices[:, 0])
                y.append(coll.vertices[:, 1])
                thelen = len(coll.vertices[:, 0])
                ci.append(thelen)
                #ci.append(coll.vertices.shape[1])
            li.append(len(cont.get_paths()))
    
        self.x = np.array([val for sublist in x for val in sublist])
        self.y = np.array([val for sublist in y for val in sublist])
        self.nb_pt_per_c = np.array(ci)
        self.ci = self.nb_pt_per_c.cumsum() - self.nb_pt_per_c
        self.nb_c_per_l = np.array(li)
        self.li = self.nb_c_per_l.cumsum() - self.nb_c_per_l
        
        self.nearesti = None
        self.level_slice = None
        
    
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
        #print self.x[self.level_slice]
        #print self.y[self.level_slice]
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
            #print self.level_view_of_contour
            indices_of_first_pts = self.ci[self.level_view_of_contour]
            for i, index_of_first_pt in enumerate(indices_of_first_pts):
                if (index_of_first_pt - indices_of_first_pts[0]) > self.nearesti:
                    return i - 1
            return i
        else:
            return False
    
    
    #def get_uavg(self, Eddy, CS, collind, centlon_e, centlat_e, poly_eff,
                #grd, eddy_radius_e, properties, save_all_uavg=False):
        #"""
        #Calculate geostrophic speed around successive contours
        #Returns the average
        
        #If save_all_uavg == True we want uavg for every contour
        #"""
        ## Unpack indices for convenience
        #imin, imax, jmin, jmax = Eddy.imin, Eddy.imax, Eddy.jmin, Eddy.jmax
        
        #points = np.array([grd.lon()[jmin:jmax, imin:imax].ravel(),
                           #grd.lat()[jmin:jmax, imin:imax].ravel()]).T
        
        ## First contour is the outer one (effective)
        #theseglon, theseglat = poly_eff.vertices[:, 0].copy(), \
                            #poly_eff.vertices[:, 1].copy()
        
        #theseglon, theseglat = eddy_tracker.uniform_resample(
            #theseglon, theseglat, method='akima')
        
        #uavg = Eddy.uspd_coeffs.ev(theseglat[:-1], theseglon[:-1]).mean()
        
        #if save_all_uavg:
            #all_uavg = [uavg]
            #pixel_min = 1 # iterate until 1 pixel
        
        #else:
            ## Iterate until PIXEL_THRESHOLD[0] number of pixels
            #pixel_min = Eddy.PIXEL_THRESHOLD[0]
        
        ##start = True
        #citer = np.nditer(CS.cvalues, flags=['c_index'])
        ##print '************************************************'
        #while not citer.finished:
            
            ### Get contour around centlon_e, centlat_e at level [collind:][iuavg]
            ##segi, poly_i = eddy_tracker.find_nearest_contour(
                            ##CS.collections[citer.index], centlon_e, centlat_e)
            
            
            ## 
            #self.swirl.set_dist_array_size(citer.index)
            #self.swirl.set_nearest_contour_index(centlon_e, centlat_e)
            #segi = Eddy.swirl.get_index_nearest_path()
            
            
            #if segi:
                
                #poly_i = CS.collections[citer.index].get_paths()[segi]
            
                ###print poly_ii is poly_i
            
            ##if poly_i is not None:
                
                
                ## 1. Ensure polygon_i is within polygon_e
                ## 2. Ensure polygon_i contains point centlon_e, centlat_e
                ## 3. Respect size range
                ##if np.all([poly_eff.contains_path(poly_i),
                        ##poly_i.contains_point([centlon_e, centlat_e]),
                        ##(mask_i_sum >= pixel_min and
                            ##mask_i_sum <= Eddy.PIXEL_THRESHOLD[1])]):
                #if poly_i.contains_point([centlon_e, centlat_e]):
                    
                    #if poly_eff.contains_path(poly_i):
                        
                        ## NOTE: contains_points requires matplotlib 1.3 +
                        #mask_i_sum = poly_i.contains_points(points).sum()
                        #if (mask_i_sum >= pixel_min and
                            #mask_i_sum <= Eddy.PIXEL_THRESHOLD[1]):
                    
                            #seglon, seglat = poly_i.vertices[:, 0], poly_i.vertices[:, 1]
                            #seglon, seglat = eddy_tracker.uniform_resample(
                                                #seglon, seglat, method='akima')
                            
                            ## Interpolate uspd to seglon, seglat, then get mean
                            #uavgseg = Eddy.uspd_coeffs.ev(seglat[:-1], seglon[:-1]).mean()
                            
                            #if save_all_uavg:
                                #all_uavg.append(uavgseg)
                            
                            #if uavgseg >= uavg:
                                #uavg = uavgseg.copy()
                                #theseglon, theseglat = seglon.copy(), seglat.copy()
                            
                            #inner_seglon, inner_seglat = seglon.copy(), seglat.copy()
                    
            #citer.iternext()
        
        #try: # Assuming interior contours have been found
            
            #cx, cy = Eddy.M(theseglon, theseglat)
            ## Speed based eddy radius (eddy_radius_s)
            #centx_s, centy_s, eddy_radius_s, junk = fit_circle(cx, cy)
            #centlon_s, centlat_s = Eddy.M.projtran(centx_s, centy_s, inverse=True)
            #if not save_all_uavg:
                #return (uavg, centlon_s, centlat_s, eddy_radius_s,
                        #theseglon, theseglat, inner_seglon, inner_seglat)
            #else:  
                #return (uavg, centlon_s, centlat_s, eddy_radius_s,
                        #theseglon, theseglat, inner_seglon, inner_seglat, all_uavg)
        
        #except Exception: # If no interior contours found, use eddy_radius_e
        
            #if not save_all_uavg:
                #return (uavg, centlon_e, centlat_e, eddy_radius_e,
                        #theseglon, theseglat, theseglon, theseglat)
            #else:
                #return (uavg, centlon_e, centlat_e, eddy_radius_e,
                        #theseglon, theseglat, theseglon, theseglat, all_uavg)

            
            