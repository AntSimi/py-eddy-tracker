# -*- coding: utf-8 -*-
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

Copyright (c) 2014-2017 by Evan Mason and Antoine Delepoulle
Email: emason@imedea.uib-csic.es
===========================================================================

tracking.py

Version 3.0.0

===========================================================================

"""
from matplotlib.dates import julian2num, num2date

from py_eddy_tracker.observations import EddiesObservations, \
    VirtualEddiesObservations, TrackEddiesObservations
from numpy import bool_, array, arange, ones, setdiff1d, zeros, uint16, \
    where, empty
from netCDF4 import Dataset
import logging


class Correspondances(list):
    """Object to store correspondances
    And run tracking
    """
    UINT32_MAX = 4294967295
    # Prolongation limit to 255
    VIRTUAL_DTYPE = 'u1'
    # ID limit to 4294967295
    ID_DTYPE = 'u4'
    # Track limit to 65535
    N_DTYPE = 'u2'

    def __init__(self, datasets, virtual=0, class_method=None, previous_correspondance=None):
        """Initiate tracking
        """
        super(Correspondances, self).__init__()
        # Correspondance dtype
        self.correspondance_dtype = [('in', 'u2'),
                                     ('out', 'u2'),
                                     ('id', self.ID_DTYPE)]
        if class_method is None:
            self.class_method = EddiesObservations
        else:
            self.class_method = class_method

        # To count ID
        self.current_id = 0
        # To know the number maximal of link between two state
        self.nb_link_max = 0
        # Dataset to iterate
        self.datasets = datasets
        self.previous2_obs = None
        self.previous_obs = None
        self.current_obs = None

        # To use virtual obs
        # Number of obs which can prolongate real observations
        self.nb_virtual = virtual
        # Activation or not
        self.virtual = virtual > 0
        self.virtual_obs = None
        self.previous_virtual_obs = None

        # Correspondance to prolongate
        self.filename_previous_correspondance = previous_correspondance
        self.previous_correspondance = self.load_compatible(self.filename_previous_correspondance)

        if self.virtual:
            # Add field to dtype to follow virtual observations
            self.correspondance_dtype += [
                # True if it isn't a real obs
                ('virtual', bool_),
                # Length of virtual segment
                ('virtual_length', self.VIRTUAL_DTYPE)]

        # Array to simply merged
        self.nb_obs_by_tracks = None
        self.i_current_by_tracks = None
        self.nb_obs = 0
        self.eddies = None

    def reset_dataset_cache(self):
        self.previous2_obs = None
        self.previous_obs = None
        self.current_obs = None

    @property
    def period(self):
        """To rethink

        Returns: period coverage by obs

        """
        date_start = num2date(julian2num(self.class_method.load_from_netcdf(self.datasets[0]).obs['time'][0] - 0.5))
        date_stop = num2date(julian2num(self.class_method.load_from_netcdf(self.datasets[-1]).obs['time'][0] - 0.5))
        return date_start, date_stop

    def swap_dataset(self, dataset):
        """ Swap to next dataset
        """
        self.previous2_obs = self.previous_obs
        self.previous_obs = self.current_obs
        self.current_obs = self.class_method.load_from_netcdf(dataset)

    def merge_correspondance(self, other):
        # Verify compliance of file
        if self.nb_virtual != other.nb_virtual:
            raise Exception('Different method of tracking')
        # Determine junction
        i = where(other.datasets == array(self.datasets[-1]))[0]
        if len(i) != 1:
            raise Exception('More than one intersection')

        # Merge
        # Create a hash table
        translate = empty(other.current_id, dtype='u4')
        translate[:] = self.UINT32_MAX

        translate[other[i[0] - 1]['id']] = self[-1]['id']

        nb_max = other[i[0] - 1]['id'].max()
        mask = translate == self.UINT32_MAX
        # We won't translate previous id
        mask[:nb_max] = False
        # Next id will be shifted
        translate[mask] = arange(mask.sum()) + self.current_id

        # Translate
        for items in other[i[0]:]:
            items['id'] = translate[items['id']]
        # Extend with other obs
        self.extend(other[i[0]:])
        # Extend datasets list, which are bounds so we add one
        self.datasets.extend(other.datasets[i[0] + 1:])
        # We set new id available
        self.current_id = translate[-1] + 1

    def store_correspondance(self, i_previous, i_current, nb_real_obs):
        """Storing correspondance in an array
        """
        # Create array to store correspondance data
        correspondance = array(i_previous, dtype=self.correspondance_dtype)
        if self.virtual:
            correspondance['virtual_length'][:] = 255
        # index from current_obs
        correspondance['out'] = i_current

        if self.virtual:
            # if index in previous dataset is bigger than real obs number
            # it's a virtual data
            correspondance['virtual'] = i_previous >= nb_real_obs

        if self.previous2_obs is None:
            # First time we set ID (Program starting)
            nb_match = i_previous.shape[0]
            # Set an id for each match
            correspondance['id'] = self.id_generator(nb_match)
            self.append(correspondance)
            return True

        # We set all id to UINT32_MAX
        id_previous = ones(len(self.previous_obs),
                           dtype=self.ID_DTYPE) * self.UINT32_MAX
        # We get old id for previously eddies tracked
        id_previous[self[-1]['out']] = self[-1]['id']
        # We store ID in correspondance if the ID is UINT32_MAX, we never
        # track it before
        correspondance['id'] = id_previous[correspondance['in']]

        # We set correspondance data for virtual obs : ID/LENGTH
        if self.previous2_obs is not None and self.virtual:
            nb_rebirth = correspondance['virtual'].sum()
            if nb_rebirth != 0:
                logging.debug('%d re-birth due to prolongation with'
                              ' virtual observations', nb_rebirth)
                ## Set id for virtual
                # get correspondance mask using virtual obs
                m_virtual = correspondance['virtual']
                # index of virtual in virtual obs
                i_virtual = correspondance['in'][m_virtual] - nb_real_obs
                correspondance['id'][m_virtual] = \
                    self.virtual_obs['track'][i_virtual]
                correspondance['virtual_length'][m_virtual] = \
                    self.virtual_obs['segment_size'][i_virtual]

        # new_id is equal to UINT32_MAX we must add a new ones
        # we count the number of new
        mask_new_id = correspondance['id'] == self.UINT32_MAX
        nb_new_tracks = mask_new_id.sum()
        logging.debug('%d birth in this step', nb_new_tracks)
        # Set new id
        correspondance['id'][mask_new_id] = self.id_generator(nb_new_tracks)

        self.append(correspondance)

        return False

    def append(self, *args, **kwargs):
        self.nb_link_max = max(self.nb_link_max, len(args[0]))
        super(Correspondances, self).append(*args, **kwargs)

    def id_generator(self, nb_id):
        """Generation id and incrementation
        """
        values = arange(self.current_id, self.current_id + nb_id)
        self.current_id += nb_id
        return values

    def recense_dead_id_to_extend(self):
        """Recense dead id to extend in virtual observation
        """
        # List previous id which are not use in the next step
        dead_id = setdiff1d(self[-2]['id'], self[-1]['id'])
        nb_dead = dead_id.shape[0]
        logging.debug('%d death of real obs in this step', nb_dead)
        if not self.virtual:
            return
        # get id already dead from few time
        nb_virtual_extend = 0
        if self.virtual_obs is not None:
            virtual_dead_id = setdiff1d(self.virtual_obs['track'],
                                        self[-1]['id'])
            list_previous_virtual_id = self.virtual_obs['track'].tolist()
            i_virtual_dead_id = [
                list_previous_virtual_id.index(i) for i in virtual_dead_id]
            # Virtual obs which can be prolongate
            alive_virtual_obs = self.virtual_obs['segment_size'
                                ][i_virtual_dead_id] < self.nb_virtual
            nb_virtual_extend = alive_virtual_obs.sum()
            logging.debug('%d virtual obs will be prolongate on the '
                          'next step', nb_virtual_extend)

        # Save previous state to count virtual obs
        self.previous_virtual_obs = self.virtual_obs
        # Creation of an virtual step for dead one
        self.virtual_obs = VirtualEddiesObservations(
            size=nb_dead + nb_virtual_extend,
            track_extra_variables=self.previous_obs.track_extra_variables,
            track_array_variables=self.previous_obs.track_array_variables,
            array_variables=self.previous_obs.array_variables)

        # Find mask/index on previous correspondance to extrapolate
        # position
        list_previous_id = self[-2]['id'].tolist()
        i_dead_id = [list_previous_id.index(i) for i in dead_id]

        # Selection of observations on N-2 and N-1
        obs_a = self.previous2_obs.obs[self[-2][i_dead_id]['in']]
        obs_b = self.previous_obs.obs[self[-2][i_dead_id]['out']]
        # Position N-2 : A
        # Position N-1 : B
        # Virtual Position : C
        # New position C = B + AB
        for key in obs_b.dtype.fields.keys():
            if key in ['lon', 'lat', 'time', 'track', 'segment_size',
                       'dlon', 'dlat'] or 'contour_' in key:
                continue
            self.virtual_obs[key][:nb_dead] = obs_b[key]
        self.virtual_obs['dlon'][:nb_dead] = obs_b['lon'] - obs_a['lon']
        self.virtual_obs['dlat'][:nb_dead] = obs_b['lat'] - obs_a['lat']
        self.virtual_obs['lon'][:nb_dead
        ] = obs_b['lon'] + self.virtual_obs['dlon'][:nb_dead]
        self.virtual_obs['lat'][:nb_dead
        ] = obs_b['lat'] + self.virtual_obs['dlat'][:nb_dead]
        # Id which are extended
        self.virtual_obs['track'][:nb_dead] = dead_id
        # Add previous virtual
        if nb_virtual_extend > 0:
            obs_to_extend = self.previous_virtual_obs.obs[i_virtual_dead_id
            ][alive_virtual_obs]
            for key in obs_b.dtype.fields.keys():
                if key in ['lon', 'lat', 'time', 'track', 'segment_size',
                           'dlon', 'dlat'] or 'contour_' in key:
                    continue
                self.virtual_obs[key][nb_dead:] = obs_to_extend[key]
            self.virtual_obs['lon'][nb_dead:
            ] = obs_to_extend['lon'] + obs_to_extend['dlon']
            self.virtual_obs['lat'][nb_dead:
            ] = obs_to_extend['lat'] + obs_to_extend['dlat']
            self.virtual_obs['track'][nb_dead:] = obs_to_extend['track']
            self.virtual_obs['segment_size'][nb_dead:
            ] = obs_to_extend['segment_size']
        # Count
        self.virtual_obs['segment_size'][:] += 1

    def load_state(self):
        # If we have a previous file of correspondance, we will replay only recent part
        if self.previous_correspondance is not None:
            first_dataset = len(self.previous_correspondance.datasets)
            for correspondance in self.previous_correspondance[:first_dataset]:
                self.append(correspondance)
            self.current_obs = self.class_method.load_from_netcdf(self.datasets[first_dataset - 2])
            flg_virtual = self.previous_correspondance.virtual
            with Dataset(self.filename_previous_correspondance) as general_handler:
                self.current_id = general_handler.last_current_id
                # Load last virtual obs
                self.virtual_obs = VirtualEddiesObservations.from_netcdf(general_handler.groups['LastVirtualObs'])
                # Load and last previous virtual obs to be merge with current => will be previous2_obs
                self.current_obs = self.current_obs.merge(
                    VirtualEddiesObservations.from_netcdf(general_handler.groups['LastPreviousVirtualObs']))
            return first_dataset, flg_virtual
        return 1, False

    def track(self):
        """Run tracking
        """
        self.reset_dataset_cache()
        first_dataset, flg_virtual = self.load_state()

        self.swap_dataset(self.datasets[first_dataset - 1])
        # We begin with second file, first one is in previous
        for file_name in self.datasets[first_dataset:]:
            self.swap_dataset(file_name)
            logging.debug('%s match with previous state', file_name)
            logging.debug('%d obs to match', len(self.current_obs))

            nb_real_obs = len(self.previous_obs)
            if flg_virtual:
                logging.debug('%d virtual obs will be add to previous',
                              len(self.virtual_obs))
                self.previous_obs = self.previous_obs.merge(self.virtual_obs)
            i_previous, i_current = self.previous_obs.tracking(
                self.current_obs)

            # return true if the first time (previous2obs is none)
            if self.store_correspondance(i_previous, i_current, nb_real_obs):
                continue

            self.recense_dead_id_to_extend()

            if self.virtual:
                flg_virtual = True

    def save(self, filename, dict_completion=None):
        self.prepare_merging()
        nb_step = len(self.datasets) - 1
        if isinstance(dict_completion, dict):
            filename = filename.format(**dict_completion)
        logging.info('Create correspondance file %s', filename)
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            # Create dimensions
            logging.debug('Create Dimensions "Nlink" : %d', self.nb_link_max)
            h_nc.createDimension('Nlink', self.nb_link_max)

            logging.debug('Create Dimensions "Nstep" : %d', nb_step)
            h_nc.createDimension('Nstep', nb_step)
            var_file_in = h_nc.createVariable(
                zlib=True, complevel=1,
                varname='FileIn', datatype='S1024', dimensions='Nstep')
            var_file_out = h_nc.createVariable(
                zlib=True, complevel=1,
                varname='FileOut', datatype='S1024', dimensions='Nstep')
            for i, dataset in enumerate(self.datasets[:-1]):
                var_file_in[i] = dataset
                var_file_out[i] = self.datasets[i + 1]

            var_nb_link = h_nc.createVariable(
                zlib=True, complevel=1,
                varname='nb_link', datatype='u2', dimensions='Nstep')

            for name, dtype in self.correspondance_dtype:
                if dtype is bool_:
                    dtype = 'u1'
                kwargs_cv = dict()
                if 'u1' in dtype:
                    kwargs_cv['fill_value'] = 255,
                h_nc.createVariable(zlib=True,
                                    complevel=1,
                                    varname=name,
                                    datatype=dtype,
                                    dimensions=('Nstep', 'Nlink'),
                                    **kwargs_cv
                                    )

            for i, correspondance in enumerate(self):
                nb_elt = correspondance.shape[0]
                var_nb_link[i] = nb_elt
                for name, _ in self.correspondance_dtype:
                    h_nc.variables[name][i, :nb_elt] = correspondance[name]
            h_nc.virtual_use = str(self.virtual)
            h_nc.virtual_max_segment = self.nb_virtual
            h_nc.last_current_id = self.current_id
            if self.virtual_obs is not None:
                group = h_nc.createGroup('LastVirtualObs')
                self.virtual_obs.to_netcdf(group)
                group = h_nc.createGroup('LastPreviousVirtualObs')
                self.previous_virtual_obs.to_netcdf(group)
            h_nc.module = self.class_method.__module__
            h_nc.classname = self.class_method.__qualname__

    def load_compatible(self, filename):
        if filename is None:
            return None
        previous_correspondance = Correspondances.load(filename)
        if self.nb_virtual != previous_correspondance.nb_virtual:
            raise Exception('File of correspondance IN contains a different virtual segment size : file(%d), yaml(%d)' %
                            (previous_correspondance.nb_virtual, self.nb_virtual))

        if self.class_method != previous_correspondance.class_method:
            raise Exception('File of correspondance IN contains a different class method: file(%s), yaml(%s)' %
                            (previous_correspondance.class_method, self.class_method))
        return previous_correspondance

    @classmethod
    def load(cls, filename):
        with Dataset(filename, 'r', format='NETCDF4') as h_nc:
            logging.info('load %s', filename)
            datasets = list(h_nc.variables['FileIn'][:])
            datasets.append(h_nc.variables['FileOut'][-1])

            if hasattr(h_nc, 'module'):
                class_method= getattr(__import__(h_nc.module, globals(), locals(), h_nc.classname), h_nc.classname)
            else:
                class_method= None
            obj = cls(datasets, h_nc.virtual_max_segment, class_method=class_method)

            id_max = 0
            for i, nb_elt in enumerate(h_nc.variables['nb_link'][:]):
                logging.debug(
                    'Link between %s and %s',
                    h_nc.variables['FileIn'][i],
                    h_nc.variables['FileOut'][i])
                correspondance = array(h_nc.variables['in'][i, :nb_elt],
                                       dtype=obj.correspondance_dtype)
                for name, _ in obj.correspondance_dtype:
                    if name == 'in':
                        continue
                    if name == 'virtual_length':
                        correspondance[name] = 255
                    correspondance[name] = h_nc.variables[name][i, :nb_elt]
                id_max = max(id_max, correspondance['id'].max())
                obj.append(correspondance)
            obj.current_id = id_max + 1
        return obj

    def prepare_merging(self):
        # count obs by tracks (we add directly one, because correspondance
        # is an interval)
        self.nb_obs_by_tracks = zeros(self.current_id, dtype=self.N_DTYPE) + 1
        for correspondance in self:
            self.nb_obs_by_tracks[correspondance['id']] += 1
            if self.virtual:
                # When start is virtual, we don't have a previous
                # correspondance
                self.nb_obs_by_tracks[
                    correspondance['id'][correspondance['virtual']]
                ] += correspondance['virtual_length'][
                    correspondance['virtual']]

        # Compute index of each tracks
        self.i_current_by_tracks = \
            self.nb_obs_by_tracks.cumsum() - self.nb_obs_by_tracks
        # Number of global obs
        self.nb_obs = self.nb_obs_by_tracks.sum()
        logging.info('%d tracks identified', self.current_id)
        logging.info('%d observations will be join', self.nb_obs)

    def merge(self, until=-1):
        """Merge all the correspondance in one array with all fields
        """
        # Start loading identification again to save in the finals tracks
        # Load first file
        self.swap_dataset(self.datasets[0])

        # Start create netcdf to agglomerate all eddy
        logging.debug('We will create an array (size %d)', self.nb_obs)
        eddies = TrackEddiesObservations(
            size=self.nb_obs,
            track_extra_variables=self.current_obs.track_extra_variables,
            track_array_variables=self.current_obs.track_array_variables,
            array_variables=self.current_obs.array_variables,
        )

        # Calculate the index in each tracks, we compute in u4 and translate
        # in u2 (which are limited to 65535)
        logging.debug('Compute global index array (N)')
        eddies['n'][:] = uint16(
            arange(self.nb_obs, dtype='u4')
            - self.i_current_by_tracks.repeat(self.nb_obs_by_tracks))
        logging.debug('Compute global track array')
        eddies['track'][:] = arange(self.current_id
                                    ).repeat(self.nb_obs_by_tracks)

        # Set type of eddy with first file
        eddies.sign_type = self.current_obs.sign_type
        # Fields to copy
        fields = self.current_obs.obs.dtype.descr

        # To know if the track start
        first_obs_save_in_tracks = zeros(self.i_current_by_tracks.shape,
                                         dtype=bool_)

        for i, file_name in enumerate(self.datasets[1:]):
            if until != -1 and i >= until:
                break
            logging.debug('Merge data from %s', file_name)
            # Load current file (we begin with second one)
            self.swap_dataset(file_name)
            # We select the list of id which are involve in the correspondance
            i_id = self[i]['id']
            # Index where we will write in the final object
            index_final = self.i_current_by_tracks[i_id]

            # First obs of eddies
            m_first_obs = ~first_obs_save_in_tracks[i_id]
            if m_first_obs.any():
                # Index in the previous file
                index_in = self[i]['in'][m_first_obs]
                # Copy all variable
                for field in fields:
                    var = field[0]
                    eddies[var][index_final[m_first_obs]
                    ] = self.previous_obs[var][index_in]
                # Increment
                self.i_current_by_tracks[i_id[m_first_obs]] += 1
                # Active this flag, we have only one first by tracks
                first_obs_save_in_tracks[i_id] = True
                index_final = self.i_current_by_tracks[i_id]

            if self.virtual:
                # If the flag virtual in correspondance is active,
                # the previous is virtual
                m_virtual = self[i]['virtual']
                if m_virtual.any():
                    # Incrementing index
                    self.i_current_by_tracks[i_id[m_virtual]
                    ] += self[i]['virtual_length'][m_virtual]
                    # Get new index
                    index_final = self.i_current_by_tracks[i_id]

            # Index in the current file
            index_current = self[i]['out']

            # Copy all variable
            for field in fields:
                var = field[0]
                eddies[var][index_final
                ] = self.current_obs[var][index_current]

            # Add increment for each index used
            self.i_current_by_tracks[i_id] += 1
            self.previous_obs = self.current_obs
        return eddies
