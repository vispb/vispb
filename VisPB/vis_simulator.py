import numpy as np
import numexpr as ne
import sys
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from astropy.coordinates import SkyCoord, Angle, EarthLocation, AltAz, TETE
from astropy.time import Time
from astropy import units
from scipy import constants
from tqdm import tqdm
from pathlib import Path
import time
from collections import OrderedDict
from multiprocessing import Process, Queue

from . import Sky_Model
from . import Beam_Model

class Vis_Simulator(object):
    
    def __init__(self, sky_model, beam_model, antenna_config,
                 ants=None, pols=['xx'], baseline_select=None, telescope_lat_lon_alt_deg=None,
                 use_TETE=False, horizon_cut=0., kx=3, ky=3, x_orientation='EAST',
                 Nthread=10, memory_limit=50, dtype_float=np.float64):
        '''
            Initialize the object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                sky_model: Sky_Model object
                    Sky_Model object that includes sky information.
                beam_model: Beam_Model object
                    Beam Model object with beam information.
                antenna_config: Dictionary
                    Dictionary contains antenna configuration about antenna numbers and antenna ENU positions.
                    Ex) antenna_config = {0: [0, 0, 0],
                                          1: [14.6, 0, 0],
                                          2: [29.2, 0, 0]}
                ants: List
                    Antenna numbers that will be used for the simulation.
                    None means all antennas in the antenna_config will be used.
                pols: List
                    Polarization array that can contain ['xx', 'yy', 'xy', 'yx'].
                baseline_select: List
                    List of tuple of antenna pairs to select baselines for visibility calculation.
                    Ex) baseline_select = [(0,1), (0,2)]
                telescope_lat_lon_alt_deg: Float
                    Latitude and longitude of the telescope in degree, and altitude in meter.
                    If None, HERA site location will be used.
                use_TETE: Boolean
                    If True, use "TETE" coordinate frame from astropy. See astropy for more details.
                    If Flase, use "icrs" coordnate frame.
                    Default is False.
                horizon_cut: Float or Str
                    Horizon definition in altitude in horizon coordinate. 0 means sharp cut at horizon.
                    File name with the horizon information can be acceptable.
                kx, ky: Int
                    Interpolation order/method at sky positions.
                    Default is 3 (cubic spline).
                x_orientation: Str
                    x_orientation for the simulation. It should be equal to beam.x_orientation.
                Nthread: Int
                    The number of thread for parallelized calculation of beam interpolation and visibility.
                memory_limit: Float
                    The maximum memory available for the calculation in GB.
                dtype_float: Str or Object
                    Data type in creating the map that can be either float32 or float64.
                    This will set corresponding dtype_complex.
                    Default is np.float64.
        '''
        
        args = locals().items()
        for key, value in args:
            if(key != 'self'):
                setattr(self, key, value)
                
        self._consistency_check()

        if(dtype_float == np.float32):
            dtype_complex = np.complex64
        elif(dtype_float == np.float64):
            dtype_complex = np.complex128
        else:
            raise TypeError('{} should be eighter np.float32 or np.float64')
        
        if(ants is None):
            ants = list(antenna_config.keys())
        else:
            self.antenna_config = {ant: antenna_config[ant] for ant in ants}
                
        bls_dict = {(ant1, ant2): np.array(antenna_config[ant2])-np.array(antenna_config[ant1])
                    for ant1 in ants
                    for ant2 in ants
                    if ant1 <= ant2}
        
        if(baseline_select is None):
            bls_use = list(bls_dict.keys())
        else:
            bls_use = baseline_select
            ants = np.unique([[bl[0], bl[1]] for bl in bls_use])
            
        for ant in ants:
            if(ant not in beam_model.which_beam.keys()):
                raise KeyError("antenna {} does not have an assigned beam".format(ant))
        ants = np.asarray(ants)
            
        bls_num = np.asarray([uvutils.antnums_to_baseline(ant1, ant2, len(ants))
                              for ant1, ant2 in bls_use])
        bls_vec = np.asarray([bls_dict[bl] for bl in bls_use])
        bls_red_grps, bls_unique_vecs, bls_len = uvutils.get_baseline_redundancies(bls_num, bls_vec)
        
        bls_red_grps = [[uvutils.baseline_to_antnums(bl, len(ants)) for bl in bls]
                        for bls in bls_red_grps]
        
        bls_ugrp_dict = {bls[0]: bls_dict[bls[0]] for bls in bls_red_grps}
        
        unique_component = np.unique(self.sky_model.component)
            
        setattr(self, 'freqs', sky_model.freqs)
        setattr(self, 'Nfreqs', sky_model.freqs.size)
        setattr(self, 'ants', ants)
        setattr(self, 'dtype_complex', dtype_complex)
        setattr(self, 'bls_dict', bls_dict)
        setattr(self, 'bls_unique_vecs', bls_unique_vecs)
        setattr(self, 'bls_red_grps', bls_red_grps)
        setattr(self, 'Nbls_unique', len(bls_red_grps))
        setattr(self, 'bls_ugrp_dict', bls_ugrp_dict)
        setattr(self, 'bls_use', bls_use)
        setattr(self, 'unique_component', unique_component)
        setattr(self, '__version__', 'v0.02')
        
        
        
    def run(self, time_array, integration_time=None, verbose=True):
        '''
            Run the visiblity calculation at each time bin center given by time_array.
            The integration_time will be calculated by
            (time_array[i]-time_array[i-1])/2 + (time_array[i+1]-time_array[i])/2.
            The integration_time of the first and last time bins are
            time_array[i+1]-time_array[i] and time_array[i]-time_array[i-1], respectively.
            If you want to save Vis_Simulator object as UVData object,
            the intergation_time should be consistent through the mock observation.
            
            Parameters:
            ----------------------------------------------------------------------------------------
                time_array: Array
                    List of times in JD.
                    The visibility calculation assumes a spontaneous observation at the time stamp
                    given by time_array.
                integration_time: Float
                    Integration time for a single time simulation.
                    Will be ignored if len(time_array) > 1 and will be automatically calculated from time_array.
                verbose: Boolean
                    Tracking the process of the visibility simulation.
                    Default is True.
        '''
        
        
        if('vis_model' in self.__dict__.keys()):
            self.__dict__.pop('vis_model', None)
            
        if isinstance(time_array, float):
            time_array = [time_array]
        time_array = np.asarray(time_array)
        lst_array = uvutils.get_lst_for_time(time_array, *self.telescope_lat_lon_alt_deg)
        lsts_deg = np.rad2deg(lst_array)
        
        if(len(time_array) == 1 and integration_time is None):
            raise KeyError("'integration_time' should be given")
        if(len(time_array) > 1):
            time_diff_sec = np.diff(time_array)*24*3600
            integration_time = np.concatenate(([time_diff_sec[0]],
                                               (time_diff_sec[:-1]+time_diff_sec[1:])*0.5,
                                               [time_diff_sec[-1]]))
        if isinstance(integration_time, int):
            integration_time = float(integration_time)
        if not isinstance(integration_time, (list, np.ndarray)):
            integration_time = [integration_time]
        
        setattr(self, 'time_array', time_array)
        setattr(self, 'integration_time', integration_time)
        setattr(self, 'lst_array', lst_array)
        setattr(self, 'lsts_deg', lsts_deg)
        setattr(self, 'verbose', verbose)
        
        
        self._estimate_memory(np.mean(time_array))
        Nfreqs_sub = self.Nfreqs_sub
        N_subband = int(np.ceil(self.Nfreqs / Nfreqs_sub))
        if(verbose == True):
            total_memory_need = self.memory_per_freq * self.Nfreqs
            if(total_memory_need < 1):
                byte_unit = 'MB'
                total_memory_need *= 1024
            else:
                byte_unit = 'GB'
            print("Given estimated total memory need {:.1f} {} and memory_limit {:.1f} GB,"\
                  .format(total_memory_need, byte_unit, self.memory_limit)+
                  " the frequency band is chuncked to {} piece(s)."\
                  .format(N_subband))
        
        # Start the simulation
        start = time.time()
        for j, lst in enumerate(lsts_deg):
            self.lsts_sub = lst
            self.Nlsts_sub = 1
            self.times_sub = time_array[j]
            if(verbose == True):
                print("")
                print('-------- LST: {:.3f} deg ({}/{}) --------'.format(lst, j+1, len(lsts_deg)))
                print("Converting ra/dec to az/alt...")
            self._radec_to_altaz(self.times_sub)
            self._get_altaz_above_horizon()
            
            if(verbose == True):
                unique_elem, occur_elem = np.unique(self.component, return_counts=True)
                unique_elem_total, occur_elem_total = np.unique(self.sky_model.component, return_counts=True)
                print("Selecting sources above the horizon")
                for component, occur in zip(unique_elem, occur_elem):
                    idx_match = np.where(unique_elem_total == component)[0][0]
                    print("  {}: {} out of {}"\
                          .format(component, occur, occur_elem_total[idx_match]))
                    
            if(verbose == True):
                print("")
                print("Beam interpolation at (az, alt) for {} beam(s)..."\
                      .format(len(self.beam_model.beam_dict.keys())))
            self._beam_interp(verbose=verbose)
            
            for i in range(N_subband):
                freqs_start = i*Nfreqs_sub
                freqs_end = np.min([(i+1)*Nfreqs_sub, self.Nfreqs])
                freqs_sub = self.freqs[freqs_start:freqs_end]
                if(verbose == True):
                    print("")
                    print('Calculate frequency subgroup ({}/{}): {:.3f} - {:.3f} MHz ({})' \
                          .format(i+1, N_subband, freqs_sub[0]*1e-6, freqs_sub[-1]*1e-6, freqs_sub.size))

                self.freqs_sub = freqs_sub

                if(verbose == True):
                    print("Starting visibility calculation...")
                self._calculate_visibility()
                
            if(j == 0):
                vis_model = self.vis_model
            else:
                for component in self.vis_model.keys():
                    for key in self.vis_model[component].keys():
                        vis_model[component][key] = np.concatenate((vis_model[component][key],
                                                                    self.vis_model[component][key]),
                                                                   axis=0)
            self.__dict__.pop('vis_model', None)
                
        self.vis_model = vis_model
                    
        end = time.time()
        if(verbose == True):
            print('Total elapse time: {:.2f}s\n'.format(end-start))
        
        
        
    def _consistency_check(self):
                
        if not np.array_equal(self.sky_model.freqs, self.beam_model.freqs):
            raise ValueError('freqs in sky_model and beam_model should be the same.')
            
        if not isinstance(self.sky_model, Sky_Model):
            raise TypeError('sky_model should be Sky_Model object.')
            
        if not isinstance(self.beam_model, Beam_Model):
            raise TypeError('beam_model should be Beam_Model object.')
            
        for beam_id in self.beam_model.beam_dict.keys():
            if self.x_orientation.upper() != self.beam_model.beam_x_orientation[beam_id].upper():
                raise ValueError("x_orientation of the beam and that of the simulation are not the same for beam_id {}.".format(beam_id))
            
        if not isinstance(self.pols, (list, np.ndarray)):
            self.pols = [self.pols]
        self.Npols = len(self.pols)
            
        for pol in self.pols:
            for beam_id in self.beam_model.beam_dict.keys():
                if(pol[0] not in self.beam_model.beam_feed[beam_id]):
                    raise KeyError("feed '{}' is not included in feed_array of beam_id {}.".format(pol[0], beam_id))
                elif(pol[1] not in self.beam_model.beam_feed[beam_id]):
                    raise KeyError("feed '{}' is not included in feed_array of beam_id {}.".format(pol[1], beam_id))

        if self.telescope_lat_lon_alt_deg is None:
            print("'telescope_lat_lon_alt_deg' is not specified. HERA site location will be used.")
            self.telescope_lat_lon_alt_deg = [-30.721526120689315, 21.428303826863015, 1051.6900000050664]
        
        if not isinstance(self.baseline_select, (type(None), list, np.ndarray)):
            raise TypeError("'baseline_select' should be a list of tuple.")
        
        
    def _estimate_memory(self, time_mean):
        if(self.dtype_complex == np.complex64):
            element_size = 8.
        elif(self.dtype_complex == np.complex128):
            element_size = 16.
        
        self._radec_to_altaz(time_mean)
        self._get_altaz_above_horizon()
        
        Nsources = self.alt.size
        Nthread = self.Nthread
        
        memory_for_beam = 0
        Npols = self.Npols
        for beam_id in self.beam_model.beam_kind.keys():
            if('analytic' in self.beam_model.beam_kind[beam_id]):
                Naxes = 1
            elif('cst' in self.beam_model.beam_kind[beam_id]):
                if(self.beam_model.beam_type[beam_id] == 'power_beam'):
                    Naxes = 1
                if(self.beam_model.beam_type[beam_id] == 'efield'):
                    Naxes = 2
            memory_for_beam += Naxes * Npols * Nsources
            
        memory_for_vis = (3*Npols+2) * Nsources * Nthread
        
        memory_per_freq = (memory_for_beam + memory_for_vis) * element_size
        
        self.memory_per_freq = memory_per_freq / (1024*1024*1024)
        self.Nfreqs_sub = int(np.floor(self.memory_limit / self.memory_per_freq))
        if(self.Nfreqs_sub < 1):
            least_memory_one_freq = 1*self.memory_per_freq
            raise ValueError('Momory limit should be larger than {} GB'.format(least_memory_one_freq))
        
        
        
    def _radec_to_altaz(self, time):
        '''
            transform equatorial coordinate to horizontal coordinate
        '''
        
        lat, lon, alt = self.telescope_lat_lon_alt_deg
        location = EarthLocation(lon = lon*units.deg, lat = lat*units.deg, height = alt*units.m)
        obstime = Time(time, format='jd', location=location)
        sky_altaz = AltAz(obstime=obstime, location=location)
        
        epoch = 'J2000'
        if self.use_TETE:
            frame = TETE(obstime=epoch)
        else:
            frame = 'icrs'
            
        ra = self.sky_model.ra
        dec = self.sky_model.dec
        sky_radec = SkyCoord(ra=ra, dec=dec, unit=('degree','degree'), frame=frame)
        
        sky_altaz = sky_radec.transform_to(sky_altaz)
        alt = sky_altaz.alt.degree
        az = sky_altaz.az.degree
        
        self.alt = alt
        self.az = az
        self.component = self.sky_model.component
    
    
    
    def _get_altaz_above_horizon(self):
        if isinstance(self.horizon_cut, (int, float)):
            idx_horizon = np.where(self.alt >= self.horizon_cut)[0]
        elif Path(self.horizon_cut).is_file:
            filename = self.horizon_cut
            f = h5py.File(filename, 'r')
            alt_horizon = interpolate.interp1d(f['horizon_azimuths'], f['horizon_profile'])(az)
            idx_horizon = np.where(self.alt >= alt_horizon)[0]
        else:
            raise TypeError("'horizon_cut' should be either a number or a file path.")

        self.alt = self.alt[idx_horizon]
        self.az = self.az[idx_horizon]
        self.component = self.component[idx_horizon]
        self.stokes_I = self.sky_model.stokes_I[:, idx_horizon]
        
        
        
    def _beam_interp(self, verbose=True):
        theta = 90 - self.alt
        phi = np.mod(90 - self.az, 360)
        
        for i, beam_id in enumerate(tqdm(self.beam_model.beam_dict.keys(), file=sys.stdout,
                                         ncols=70, nrows=4, colour='GREEN', disable=not verbose)):
            self.beam_model.evaluate(beam_id, phi, theta,
                                     Nthread=self.Nthread, kx=self.kx, ky=self.ky)
    
    
    
    def _fringe_calc(self, u, v, w, alt, az, phase, exp_phase):
        '''
            Calculate fringe
        '''
        
        alt = np.deg2rad(alt)[np.newaxis,:].astype(self.dtype_float)
        az = np.deg2rad(az)[np.newaxis,:].astype(self.dtype_float)
        
        l = np.zeros((1, alt.shape[-1]), dtype=self.dtype_float)
        m = np.zeros((1, alt.shape[-1]), dtype=self.dtype_float)
        n = np.zeros((1, alt.shape[-1]), dtype=self.dtype_float)
        
        tpi = self.dtype_float(2*np.pi)
        ne.evaluate('cos(alt)*sin(az)', out = l)
        ne.evaluate('cos(alt)*cos(az)', out = m)
        ne.evaluate('sin(alt)', out = n)
        
        ne.evaluate('l * u + m * v + n * w', out = phase)
        
        ne.evaluate('complex(cos(tpi*phase), -sin(tpi*phase))', out = exp_phase, casting='same_kind')
        
        
                
    def _assign_jobs_to_core(self, rank):
        '''
            Get job info to be assigned to each thread
        '''

        Nthread = self.Nthread
        N_jobs = self.Nbls_unique
        indices = np.arange(N_jobs)

        job_info = {}
        job_info['rank'] = rank
        job_info['bls_index'] = indices[rank::Nthread]

        return job_info
        
            
            
    def _calculate_visibility(self):
        '''
             Calculate visibilites using multiprocessing

             vis_model = (N_source_type, Nlsts, Nfreqs)
        '''
        
        Nthread = self.Nthread
        queue = Queue()
        for rank in range(Nthread):
            p = Process(target=self._calculate_visibility_single_thread, args=(rank, queue))
            p.start()
        
        vis_unsorted = [queue.get() for rank in range(Nthread)]

        queue.close()
        queue.join_thread()

        for rank in range(Nthread):
            for i in range(Nthread):
                if(vis_unsorted[i][0] == rank):
                    if(rank == 0):
                        vis_dict = vis_unsorted[i][1]
                    else:
                        if(bool(vis_unsorted[i][1]) is True):
                            vis_dict = {**vis_dict, **vis_unsorted[i][1]}

        vis_model = OrderedDict()
        for i, component in enumerate(self.unique_component):
            vis_comp = OrderedDict()
            for bl in self.bls_use:
                for pol in self.pols:
                    key = (bl + (pol,))
                    vis_comp[key] = vis_dict[key][i]
            vis_model[component] = vis_comp

        if('vis_model' not in self.__dict__.keys()):
            self.vis_model = vis_model
        else:
            for component in self.vis_model.keys():
                for key in self.vis_model[component].keys():
                    self.vis_model[component][key] = np.concatenate((self.vis_model[component][key],
                                                                     vis_model[component][key]),
                                                                    axis=1)
        
    
    
    def _calculate_visibility_single_thread(self, rank, queue):
        '''
            Calculate visibilities for each thread
        '''
        
        freqs_sub = self.freqs_sub.astype(self.dtype_float)
        Nfreqs_sub = freqs_sub.size
        idx_freqs = np.asarray([np.where(self.freqs == freq)[0][0] for freq in freqs_sub])
        Npols = self.Npols
        
        N_sources = len(self.component)
        unique_component = self.unique_component
        
        # uvw setup for unique baselines
        bls_unique = self.bls_ugrp_dict
        keys_unique = list(bls_unique.keys())
        wv = constants.c / freqs_sub
        bl_x = np.array(list(bls_unique.values()))[:,0]
        bl_y = np.array(list(bls_unique.values()))[:,1]
        bl_z = np.array(list(bls_unique.values()))[:,2]
        u_unique = bl_x[:, np.newaxis, np.newaxis] / wv[np.newaxis, :, np.newaxis]
        v_unique = bl_y[:, np.newaxis, np.newaxis] / wv[np.newaxis, :, np.newaxis]
        w_unique = bl_z[:, np.newaxis, np.newaxis] / wv[np.newaxis, :, np.newaxis]
        
        # Setting up the jobs to each thread
        job_info = self._assign_jobs_to_core(rank)
        
        # Sky model at local frame
        alt = self.alt
        az = self.az
        flux_density = self.stokes_I[np.newaxis, idx_freqs, :] * 0.5 # follow the stokse-I convention for unpolarized light
        
        phase = np.zeros((Nfreqs_sub, N_sources), dtype=self.dtype_float)
        exp_phase = np.zeros((Nfreqs_sub, N_sources), dtype=self.dtype_complex)
        flux_fringe = np.zeros((Npols, Nfreqs_sub, N_sources), dtype=self.dtype_complex)
        power_beam = np.zeros((Npols, Nfreqs_sub, N_sources), dtype=self.dtype_complex)
        power_beam_per_pol = np.zeros((Nfreqs_sub, N_sources), dtype=self.dtype_complex)
        vis_all = np.zeros((Npols, Nfreqs_sub, N_sources), dtype=self.dtype_complex)
        vis_dict = {}
        
        if(rank == 0):
            disable = not self.verbose
        elif(rank != 0):
            disable = True
        for i, idx_bls in enumerate(tqdm(job_info['bls_index'], file=sys.stdout,
                                         ncols=70, nrows=4, colour='BLUE', disable=disable)):
            # Calculate fringe (u_unique = (Nbls, Nfreqs_sub, N_sources))
            self._fringe_calc(u_unique[idx_bls,:,:], v_unique[idx_bls,:,:], w_unique[idx_bls,:,:],
                              alt, az, phase, exp_phase)
            # fringe = (Npols, Nfreqs_sub, N_sources)
            fringe = exp_phase[np.newaxis,:,:]

            ne.evaluate('flux_density * fringe', out = flux_fringe, casting='same_kind')
            
            key = keys_unique[idx_bls]
            idx_grp = np.where([key in bls for bls in self.bls_red_grps])[0][0]
            redundant_vis = {}
            for bl in self.bls_red_grps[idx_grp]:
                beam_id1 = self.beam_model.which_beam[bl[0]]
                beam_id2 = self.beam_model.which_beam[bl[1]]
                if((beam_id1, beam_id2) not in redundant_vis.keys()):
                    if((self.beam_model.beam_type[beam_id1] == 'efield') and
                       (self.beam_model.beam_type[beam_id2] == 'efield')):
                        for p, pol in enumerate(self.pols):
                            idx_p1 = np.where(self.beam_model.beam_feed[beam_id1] == pol[0])[0][0]
                            idx_p2 = np.where(self.beam_model.beam_feed[beam_id2] == pol[1])[0][0]
                            beam1 = self.beam_model.beam_intp[beam_id1][:,idx_p1,idx_freqs,:]
                            beam2 = self.beam_model.beam_intp[beam_id2][:,idx_p2,idx_freqs,:]
                            ne.evaluate('sum(beam1*conj(beam2), axis=0)', out=power_beam_per_pol)
                            power_beam[p] = power_beam_per_pol
                            
                    elif(self.beam_model.beam_type[beam_id1] == 'power_beam'):
                        if(beam_id1 != beam_id2):
                            raise KeyError("beam_id1 and beam_id2 should be the same when beam_type = 'power_beam'")
                        for p, pol in enumerate(self.pols):
                            idx_p = np.where(self.beam_model.beam_pol[beam_id1] == pol)[0][0]
                            power_beam[p] = self.beam_model.beam_intp[beam_id1][0,idx_p,idx_freqs,:]
                            
                    else:
                        raise ValueError("beam_type of beam_id1 and beam_id2 should be the same.")

                    # power_beam = (Nfreqs_sub, N_sources, Npols)
                    ne.evaluate('flux_fringe * power_beam', out = vis_all, casting='same_kind')
                    redundant_vis[(beam_id1, beam_id2)] = vis_all
                else:
                    vis_all = redundant_vis[(beam_id1, beam_id2)]

                vis = []
                for component in unique_component:
                    idx_match = np.where(self.component == component)[0]
                    # vis_temp = (Npols, Nfreqs_sub)
                    vis_temp = np.sum(vis_all[:, :, idx_match], axis=2, dtype=self.dtype_complex)

                    vis.append(vis_temp)
                # vis = (Nsource_type, Npols, Nfreqs_sub)
                vis = np.asarray(vis)
                for ipol, pol in enumerate(self.pols):
                    vis_dict[bl + (pol,)] = vis[:, ipol, np.newaxis, :]
                
        queue.put([rank, vis_dict])

        
        
    def get_UVData(self, component, telescope_name='HERA', history=None):
        '''
            Get UVData object from Vis_Simulator object.
            
            Parameters:
            ----------------------------------------------------------------------------------------
                component: List
                    List of string of source components to be converted in UVData object format.
                    With elements more than 1, the visibility of multiple source components
                    will be added together.
        '''
        if not isinstance(component, (list, np.ndarray)):
            component = [component]

        uvd = UVData()

        uvd.telescope_location_lat_lon_alt_degrees = self.telescope_lat_lon_alt_deg
        uvd.telescope_name = telescope_name
        uvd.instrument = 'simulator'
        if(history is None):
            uvd.history = 'simulation by VisPB {}'.format(self.__version__)
        else:
            uvd.history = history
        uvd.antenna_numbers = self.ants
        uvd.antenna_names = ['HH'+str(ant) for ant in self.ants]
        uvd.object_name = 'zenith'
        uvd.phase_type = 'drift'
        uvd.Nants_data = self.ants.size
        uvd.Nants_telescope = self.ants.size
        uvd.polarization_array = np.array([uvutils.polstr2num(pol) for pol in self.pols], dtype=np.int32)
        uvd.Npols = self.Npols
        uvd.spw_array = np.array([0])
        uvd.Nspws = uvd.spw_array.size
        uvd.freq_array = self.freqs[np.newaxis,:]
        uvd.Nfreqs = uvd.freq_array.shape[1]
        if(uvd.Nfreqs > 1):
            uvd.channel_width = np.diff(uvd.freq_array[0])[0]
        else:
            uvd.channel_width = 122070.3125
        uvd.antenna_diameters = 14.+np.zeros(len(uvd.antenna_numbers))
        uvd.vis_units = 'Jy'
        uvd.Nbls = len(self.bls_use)
        uvd.x_orientation = self.x_orientation
        lat, lon, alt = uvd.telescope_location_lat_lon_alt_degrees
        ant_pos = self.antenna_config
        ant_pos_select = [ant_pos[ant] for ant in self.ants]
        uvd.antenna_positions = uvutils.ECEF_from_ENU(ant_pos_select, np.radians(lat), np.radians(lon), alt) \
                              - uvd.telescope_location
        
        uvd.Ntimes = self.time_array.size
        uvd.Nblts = uvd.Nbls*uvd.Ntimes
        uvd.time_array = np.repeat(self.time_array, uvd.Nbls)
        uvd.times = np.unique(uvd.time_array)
        uvd.lst_array = uvutils.get_lst_for_time(uvd.time_array, lat, lon, alt)
        uvd.integration_time = np.repeat(self.integration_time, uvd.Nbls)
        ant_1_array, ant_2_array = [], []
        baseline_array = []
        uvw_array = []
        for bl in self.bls_use:
            ant1, ant2 = bl
            ant_1_array.append(ant1)
            ant_2_array.append(ant2)
            baseline_array.append(uvd.antnums_to_baseline(ant1,ant2))
            uvw_array.append(list(np.array(ant_pos[ant2])-np.array(ant_pos[ant1])))
        uvd.ant_1_array = np.tile(ant_1_array, uvd.Ntimes)
        uvd.ant_2_array = np.tile(ant_2_array, uvd.Ntimes)
        uvd.baseline_array = np.tile(baseline_array, uvd.Ntimes)
        uvd.uvw_array = np.array([np.tile(np.array(uvw_array)[:,i], uvd.Ntimes) for i in range(3)]).T
        uvd.data_array = np.zeros((uvd.Nblts,uvd.Nspws,uvd.Nfreqs,uvd.Npols), dtype=self.dtype_complex)
        uvd.flag_array = np.zeros((uvd.Nblts,uvd.Nspws,uvd.Nfreqs,uvd.Npols), dtype=bool)
        uvd.nsample_array = np.ones((uvd.Nblts,uvd.Nspws,uvd.Nfreqs,uvd.Npols), dtype=float)

        for ipol, pol in enumerate(self.pols):
            for bl in self.bls_use:
                ant1, ant2 = bl
                inds = uvd.antpair2ind(ant1, ant2)
                key = (ant1, ant2, pol)
                data_blts = 0
                for comp in component:
                    data_blts += self.vis_model[comp][key][:,:]
                uvd.data_array[inds,0,:,ipol] = data_blts

        return uvd
