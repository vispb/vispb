import numpy as np
import os
import copy
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils
import scipy
from scipy import interpolate
from astropy import modeling
from astropy import constants
from pathlib import Path
from multiprocessing import Queue, Process
from .data import DATA_PATH


class Beam_Model(object):
    
    def __init__(self, freqs, dtype_float=np.float64):
        '''
            Initialize the object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                freqs: Array
                    Frequency where the beam is evaluated in Hz
                dtype_float: Str or Object
                    Data type in creating the map that can be either float32 or float64
                    This will set corresponding dtype_complex
                    Default is np.float64
        '''
        
        if isinstance(freqs, (list, np.ndarray)) is False:
            freqs = [freqs]
        elif isinstance(freqs, tuple):
            raise TypeError('freqs should be either list or numpy array')
            
        freqs = np.asarray(freqs)
        
        if(dtype_float == np.float32):
            dtype_complex = np.complex64
        elif(dtype_float == np.float64):
            dtype_complex = np.complex128
        else:
            raise TypeError('{} should be either np.float32 or np.float64')
            
        setattr(self, 'freqs', freqs)
        setattr(self, 'dtype_float', dtype_float)
        setattr(self, 'dtype_complex', dtype_complex)        
        setattr(self, 'beam_dict', {})
        setattr(self, 'which_beam', {})
        setattr(self, 'beam_kind', {})
        setattr(self, 'beam_type', {})
        setattr(self, 'beam_intp', {})
        setattr(self, 'beam_feed', {})
        setattr(self, 'beam_pol', {})
        setattr(self, 'beam_x_orientation', {})
        
    
    
    def read_uniform_beam(self, beam_id, ant_id, mask_phi=None, mask_theta=None, beam_type='efield'):
        '''
            Read uniform or tophat or binary beams (1 for unmasked region, 0 for otherwise)
            if mask_za and/or mask_az are given
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                ant_id: Int or Array
                    Antenna ID that the beam is assigned to.
                    If more than one antenna ID are given, the same beam model will be assigned to all given antennas.
                mask_phi: Array or Tuple
                    phi angle range to be maksed in degree
                mask_theta: Array or Tuple
                    theta angle range to be masked in degree (e.g., if mask_za = [10, 20], 10 =< za =< 20 will be masked)
        '''
        
        if isinstance(ant_id, int):
            ant_id = np.array([ant_id])
        elif isinstance(ant_id, (list, np.ndarray)):
            ant_id = np.asarray(ant_id)
        else:
            raise TypeError('ant_id should be either integer or array')
            
        for ant_check in ant_id:
            if(ant_check in self.which_beam.keys()):
                raise ValueError('antenna {} already has a beam assigned. Each antenna should have only one beam.'.format(ant_check))
                
        if(beam_id in self.beam_dict.keys()):
            raise ValueError('beam_id: {} is already occupied. Pick other beam_id'.format(beam_id))
            
        def beam_function(phi, theta, mask_phi=mask_phi, mask_theta=mask_theta):
            '''
                Analytic beam model for the uniform response. Evaluate at (phi, theta) when it is called

                Parameters:
                ----------------------------------------------------------------------------------------
                    phi: Array
                        phi in degree. 1D array that should have the same length of theta
                    thata: Array
                        theta in degree. 1D array that should have the same length of phi
            '''
            
            if isinstance(phi, (int, float)):
                phi = [phi]
            if isinstance(theta, (int, float)):
                theta = [theta]
            phi = np.asarray(phi)
            theta = np.asarray(theta)

            freqs = self.freqs
            beam = np.ones((freqs.size, theta.size), dtype=self.dtype_float)
            
            if(mask_theta is not None):
                mask_theta = np.sort(mask_theta)
                idx_mask = np.where(np.logical_and(theta >= mask_theta[0], theta <= mask_theta[1]))[0]
                beam[:, idx_mask] = 0
                
            if(mask_phi is not None):
                mask_phi = np.sort(mask_phi)
                idx_mask = np.where(np.logical_and(phi >= mask_phi[0], phi <= mask_phi[1]))[0]
                beam[:, idx_mask] = 0
            
            return beam[np.newaxis, np.newaxis, :, :]
        
        self.beam_dict[beam_id] = beam_function
        self.beam_kind[beam_id] = 'analytic uniform beam'
        self.beam_type[beam_id] = 'efield'
        self.beam_feed[beam_id] = np.array(['x'])
        self.beam_x_orientation[beam_id] = 'EAST'
        for ant_id_ in ant_id:
            self.which_beam[ant_id_] = beam_id
    
    
    
    def read_gaussian_beam(self, beam_id, ant_id, sigma0, nu0=1e8, alpha=0.0, beam_type='efield'):
        '''
            Read chromatic symmetric Gaussian beam that is normalized to the peak
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                ant_id: Int or Array
                    Antenna ID that the beam is assigned to.
                    If more than one antenna ID are given, the same beam model will be assigned to all given antennas.
                sigma0: Float
                    Standard deviation at nu0 for the chromatic Gaussian beam in degree.
                    If alpha=0, sigma0 will be a constant standard deviation.
                nu0: Float
                    Zero-point frequency to define chromatic Gaussian beam in Hz. sigma = sigma0*(freqs/nu0)^alpha
                alpha: Float
                    Power to define the chromatic Gaussian beam. If alpha=0, the beam is achromatic.
                    
        '''
        
        if isinstance(ant_id, int):
            ant_id = np.array([ant_id])
        elif isinstance(ant_id, (list, np.ndarray)):
            ant_id = np.asarray(ant_id)
        else:
            raise TypeError('ant_id should be either integer or array')
            
        for ant_check in ant_id:
            if(ant_check in self.which_beam.keys()):
                raise ValueError('antenna {} already has a beam assigned. Each antenna should have only one beam.'.format(ant_check))
                
        if(beam_id in self.beam_dict.keys()):
            raise ValueError('beam_id: {} is already occupied. Pick other beam_id'.format(beam_id))
            
        def beam_function(phi, theta, sigma0=sigma0, nu0=nu0, alpha=0):
            '''
                Analytic beam model for the Gaussian beam response. Evaluate at (phi, theta) when it is called

                Parameters:
                ----------------------------------------------------------------------------------------
                    phi: Array
                        phi in degree. 1D array that should have the same length of theta
                    thata: Array
                        theta in degree. 1D array that should have the same length of phi
            '''
            
            if isinstance(phi, (int, float)):
                phi = [phi]
            if isinstance(theta, (int, float)):
                theta = [theta]
            phi = np.asarray(phi)
            theta = np.asarray(theta)
            
            freqs = self.freqs
            sigma = sigma0*(freqs/nu0)**alpha
            beam = np.exp(-theta[np.newaxis,:]**2/(2*sigma[:,np.newaxis]**2))
        
            return beam[np.newaxis, np.newaxis, :, :]

        self.beam_dict[beam_id] = beam_function
        self.beam_kind[beam_id] = 'analytic gaussian beam'
        self.beam_type[beam_id] = 'efield'
        self.beam_feed[beam_id] = np.array(['x'])
        self.beam_x_orientation[beam_id] = 'EAST'
        for ant_id_ in ant_id:
            self.which_beam[ant_id_] = beam_id
        
    
    
    def read_airy_beam(self, beam_id, ant_id, x_pc=0.0, y_pc=0.0, a_x=6.0, a_y=6.0):
        '''
            Read more generalized chromatic airy beam that is normalized to the peak.
            The airy beam is defined by B = |2*jv(1,arg)/arg|
            where jv is Bessel function of the first kind with the first order (jv(1,x))
            and arg = k*sqrt({a_x*(x-x_pc)}^2+{a_y*(y-y_pc)}^2), k = 2pi/wavelength
            and x, y are the unit spherical coordinate (x = sin(theta)*cos(phi), y = sin(theta)*sin(phi))
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                ant_id: Int or Array
                    Antenna ID that the beam is assigned to.
                    If more than one antenna ID are given, the same beam model will be assigned to all given antennas.
                x_pc, y_pc: Float
                    Pointing center for the airy beam defined in the unit sphere coordinate.
                a_x, a_y: Float
                    Semi-major/minor axis in meter: Default is 6 m.
                    
        '''
        
        if isinstance(ant_id, int):
            ant_id = np.array([ant_id])
        elif isinstance(ant_id, (list, np.ndarray)):
            ant_id = np.asarray(ant_id)
        else:
            raise TypeError('ant_id should be either integer or array')
            
        for ant_check in ant_id:
            if(ant_check in self.which_beam.keys()):
                raise ValueError('antenna {} already has a beam assigned. Each antenna should have only one beam.'.format(ant_check))
                
        if(beam_id in self.beam_dict.keys()):
            raise ValueError('beam_id: {} is already occupied. Pick other beam_id'.format(beam_id))

        def beam_function(phi, theta, x_pc=x_pc, y_pc=y_pc, a_x=a_x, a_y=a_y):
            '''
                Analytic beam model for the airy beam response. Evaluate at (phi, theta) when it is called

                Parameters:
                ----------------------------------------------------------------------------------------
                    phi: Array
                        phi in degree. 1D array that should have the same length of theta
                    thata: Array
                        theta in degree. 1D array that should have the same length of phi
            '''
            
            if isinstance(phi, (int, float)):
                phi = [phi]
            if isinstance(theta, (int, float)):
                theta = [theta]
            phi = np.asarray(phi)
            theta = np.asarray(theta)
            
            freqs = self.freqs
            x = (np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)))[np.newaxis, :]
            y = (np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)))[np.newaxis, :]

            k = 2*np.pi * freqs / constants.c.value
            k = k[:, np.newaxis]

            arg = k*np.sqrt(a_x**2*(x-x_pc)**2+a_y**2*(y-y_pc)**2)

            beam = np.abs(2*scipy.special.jv(1,arg)/arg)
            beam = np.where(np.isnan(beam), 1, beam)
        
            return beam[np.newaxis, np.newaxis, :, :]

        self.beam_dict[beam_id] = beam_function
        self.beam_kind[beam_id] = 'analytic airy beam'
        self.beam_type[beam_id] = 'efield'
        self.beam_feed[beam_id] = np.array(['x'])
        self.beam_x_orientation[beam_id] = 'EAST'
        for ant_id_ in ant_id:
            self.which_beam[ant_id_] = beam_id

            
            
    def read_airy_beam2(self, beam_id, ant_id, diameter=12.0, amplitude=1.0, x0=0.0, y0=0.0):
        '''
            Read chromatic airy beam that is normalized to the peak.
            The airy beam is called from astropy.modeling.functional_models.AiryDisk2D
            and will be evaluated at (x, y), unit spherical coordinate
            (x = sin(theta)*cos(phi), y = sin(theta)*sin(phi))
            beam_type is forced to "power_beam"
            This second type of airy_beam is added for the direct comparison with
            direct_optimal_mapping code mainly developed by Zhilei Xu.
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                ant_id: Int or Array
                    Antenna ID that the beam is assigned to.
                    If more than one antenna ID are given, the same beam model will be assigned to all given antennas.
                diameter: Float
                    Diameter of the aperture in meter. Default is 12.0 m for the underilluminated HERA dish
                amplitude: Float
                    Amplitude of the beam. Default is 1.
                x0, y0: Float
                    Pointing center for the airy beam.
                    
        '''
        
        if isinstance(ant_id, int):
            ant_id = np.array([ant_id])
        elif isinstance(ant_id, (list, np.ndarray)):
            ant_id = np.asarray(ant_id)
        else:
            raise TypeError('ant_id should be either integer or array')
            
        for ant_check in ant_id:
            if(ant_check in self.which_beam.keys()):
                raise ValueError('antenna {} already has a beam assigned. Each antenna should have only one beam.'.format(ant_check))
                
        if(beam_id in self.beam_dict.keys()):
            raise ValueError('beam_id: {} is already occupied. Pick other beam_id'.format(beam_id))

        def beam_function(phi, theta, diameter=diameter, amplitude=amplitude, x0=x0, y0=y0):
            '''
                Analytic beam model for the airy beam response. Evaluate at (phi, theta) when it is called

                Parameters:
                ----------------------------------------------------------------------------------------
                    phi: Array
                        phi in degree. 1D array that should have the same length of theta
                    thata: Array
                        theta in degree. 1D array that should have the same length of phi
            '''
            
            if isinstance(phi, (int, float)):
                phi = [phi]
            if isinstance(theta, (int, float)):
                theta = [theta]
            phi = np.asarray(phi)
            theta = np.asarray(theta)
            
            freqs = self.freqs
            x = (np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(phi)))[np.newaxis, :]
            y = (np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi)))[np.newaxis, :]

            R = 1.22 * constants.c.value / freqs / diameter
            beam = modeling.functional_models.AiryDisk2D(amplitude=amplitude,
                                                         x_0=x0,
                                                         y_0=y0,
                                                         radius=R[:, np.newaxis])

            return beam(x, y)[np.newaxis, np.newaxis, :, :]

        self.beam_dict[beam_id] = beam_function
        self.beam_kind[beam_id] = 'analytic airy beam2'
        self.beam_type[beam_id] = 'power_beam'
        self.beam_feed[beam_id] = np.array(['x'])
        self.beam_pol[beam_id] = np.array(['xx'])
        self.beam_x_orientation[beam_id] = 'EAST'
        for ant_id_ in ant_id:
            self.which_beam[ant_id_] = beam_id
        
        
        
    def read_cst_beam(self, beam_id, ant_id, beam_filename,
                      feed=None, horizon_below_inds=None, efield_to_power=False,
                      interp_kind='cubic', set_x_orientation=None):
        '''
            Read CST-simulated beam in UVBeam format.
            The beam will be interpolated at the given frequency chennel
            using scipy.interpolate.interp1d with the given interpolation method (interp_kind).
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                ant_id: Int or Array
                    Antenna ID that the beam is assigned to.
                    If more than one antenna ID are given, the same beam model will be assigned to all given antennas.
                beam_filename: Str or UVBeam object
                    Full path to the beam file simulated by CST or UVBeam object
                feed: Str
                    If specified, 'x' or 'y' is allowed. None means select all feeds available in the beamfits file.
                horizon_below_inds: Int
                    If set, only use theta-axis array index to horizon (theta <= 90) + horizon_below_inds for buffer
                    (i.e., 90 deg + extra horizon buffer for better beam interpolation at the edge).
                    This speeds up the interpolation.
                    If none, use all theta-axis elements for beam interpolation
                efield_to_power: Boolean
                    Wheter convert efield to power beam or not. Only can be used when beam_type = 'efield'.
                    Default is False.
                interp_kind: Str
                    Interpolation kind along the frequency with scipy.interpolate.interp1d.
                    Default is 'cubic'.
                set_x_orientation: Str
                    When the input beam object does not specify the 'x_orientation' and we know the 'x_orientation',
                    'set_x_orientation' can be used to set the 'x_orientation'.
                    For visibility simulations, this should be specified.
                    For HERA, dipole feed has x_orientation = 'EAST', while vivaldi feed has x_orientation = 'NORTH'.
        '''
        
        if isinstance(ant_id, int):
            ant_id = np.array([ant_id])
        elif isinstance(ant_id, (list, np.ndarray)):
            ant_id = np.asarray(ant_id)
        else:
            raise TypeError('ant_id should be either integer or array')
            
        for ant_check in ant_id:
            if(ant_check in self.which_beam.keys()):
                raise ValueError('antenna {} already has a beam assigned. Each antenna should have only one beam.'.format(ant_check))
                
        if(beam_id in self.beam_dict.keys()):
            raise ValueError('beam_id: {} is already occupied. Pick other beam_id'.format(beam_id))
        
        if isinstance(beam_filename, str):
            uvb = UVBeam()
            uvb.read_beamfits(beam_filename)
        elif isinstance(beam_filename, UVBeam):
            uvb = copy.deepcopy(beam_filename)
        else:
            raise TypeError('beam_filename should be either beam_file path or UVBeam object')
            
        if(feed is not None):
            if(feed not in uvb.feed_array):
                raise ValueError("feed {} is not available in the beam object".format(feed))
            else:
                uvb.select(feeds=feed)
        self.beam_feed[beam_id] = uvb.feed_array
        
        self.horizon_below_inds = horizon_below_inds
        if(horizon_below_inds is not None):
            axis2_array = np.rad2deg(uvb.axis2_array)
            idx_horizon = np.where(axis2_array >= 90)[0]
            uvb.select(axis2_inds=np.arange(0, idx_horizon.size+horizon_below_inds))
            
        if(uvb.x_orientation is None and set_x_orientation is None):
            raise KeyError("x_orientation of the beam is not set. Define it with the argument 'set_x_orientation'")
        elif(uvb.x_orientation is None and set_x_orientation is not None):
            self.beam_x_orientation[beam_id] = set_x_orientation
        elif(uvb.x_orientation is not None and set_x_orientation is not None):
            print("beam object already has x_orientation={}. Ignore set_x_orientation.".format(uvb.x_orientation))
            self.beam_x_orientation[beam_id] = uvb.x_orientation
        else:
            self.beam_x_orientation[beam_id] = uvb.x_orientation
            
            
        if(uvb.Nspws != 1):
            raise ValueError("Only single spectral window (Nspws=1) is acceptable by the simulator.")
            
        if(efield_to_power == True):
            uvb.efield_to_power()
            self.beam_type[beam_id] = 'power_beam'
            self.beam_pol[beam_id] = np.array(uvutils.polnum2str(uvb.polarization_array))
        else:
            self.beam_type[beam_id] = 'efield'

        uvb.peak_normalize()
            
        uvb.freq_interp_kind = interp_kind
        uvb.interpolation_function = 'az_za_simple'
        uvb = uvb.interp(freq_array=self.freqs, new_object=True, check_azza_domain=False)

        self.beam_dict[beam_id] = uvb
        self.beam_kind[beam_id] = 'cst simulated beam: {}'.format(os.path.basename(beam_filename))

        for ant_id_ in ant_id:
            self.which_beam[ant_id_] = beam_id
            
            
            
    def reset_beam_id(self, beam_id):
        '''
            Remove the beam_id from Beam_Model object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    beam_id to be removed
        '''
        
        self.beam_dict.pop(beam_id)
        self.beam_kind.pop(beam_id)
        self.which_beam = {
            ant_id: beam_id_val for ant_id, beam_id_val in self.which_beam.items()
            if beam_id_val != beam_id
        }
        
        
        
    def evaluate(self, beam_id, phi, theta, Nthread=10, kx=3, ky=3):
        '''
            Evaluate beam at given (phi, theta).
            For analytic beam, it computes the function values at the sky positions.
            For cst simulated beam, it interpolates the beam at the sky positions.
            
            Parameters:
            ----------------------------------------------------------------------------------------
                beam_id: Int
                    Unique ID number of the beam
                phi: Array
                    phi in degree. 1D array that should have the same length of theta.
                thata: Array
                    theta in degree. 1D array that should have the same length of phi.
                Nthread: Int
                    The number of threads to be used for parallelized interpolation of beams at (phi, theta).
                    Only applicable to cst simulated beams.
                kx, ky: Int
                    The order of interpolation for scipy.interpolate.RectBivariateSpline.
                    Default is 3.
        '''
        
        if isinstance(phi, (int, float)):
            phi = [phi]
        if isinstance(theta, (int, float)):
            theta = [theta]
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        if('analytic' in self.beam_kind[beam_id]):
            self.beam_intp[beam_id] = self.beam_dict[beam_id](phi, theta)
        elif('cst' in self.beam_kind[beam_id]):
            self.beam_intp[beam_id] = self._interp(beam_id, phi, theta, Nthread, kx, ky)
        else:
            raise ValueError("beam_kind should include either 'analytic' or 'cst'")
    
    
    
    def _interp(self, beam_id, phi, theta, Nthread, kx, ky):
        queue = Queue()
        for rank in range(Nthread):
            p = Process(target=self._interp_single_thread, args=(rank, queue, Nthread,
                                                                 beam_id, phi, theta, kx, ky))
            p.start()

        beam_collect = [queue.get() for rank in range(Nthread)]
        
        queue.close()
        queue.join_thread()

        for rank in range(Nthread):
            for i in range(Nthread):
                if(beam_collect[i][0] == rank):
                    if(rank == 0):
                        beam_intp = beam_collect[i][1]
                    else:
                        if(len(beam_collect[i][1]) != 0):
                            beam_intp = np.concatenate((beam_intp, beam_collect[i][1]), axis=-1)

        return beam_intp
            
            
            
    def _interp_single_thread(self, rank, queue, Nthread, beam_id, phi, theta, kx, ky):
        
        uvb = self.beam_dict[beam_id]                
        Naxes_vec, Nspws, Npols, Nfreqs, _, _ = uvb.data_array.shape

        phi_beam, beam_data = self._extend_beam_in_phi(uvb)
        theta_beam = np.rad2deg(uvb.axis2_array)
        beam_data = beam_data[:, 0]
        
        N_jobs_each_thread = int(np.ceil(len(phi) / Nthread))
        for i in range(Nthread):
            if(i == rank):
                phi_sub = phi[i*N_jobs_each_thread:(i+1)*N_jobs_each_thread]
                theta_sub = theta[i*N_jobs_each_thread:(i+1)*N_jobs_each_thread]
                beam_array = np.zeros((Naxes_vec, Npols, Nfreqs, len(phi_sub)), dtype=self.dtype_complex)
                for v in range(Naxes_vec):
                    for p in range(Npols):
                        beam_real = np.array(list(map(
                            lambda z: interpolate.RectBivariateSpline(theta_beam, phi_beam, z, kx=kx, ky=ky)\
                                                      (theta_sub, phi_sub, grid=False),
                            beam_data[v,p,:,:,:].real
                        )))
                        if(self.beam_type[beam_id] == 'efield'):
                            beam_imag = np.array(list(map(
                                lambda z: interpolate.RectBivariateSpline(theta_beam, phi_beam, z, kx=kx, ky=ky)\
                                                          (theta_sub, phi_sub, grid=False),
                                beam_data[v,p,:,:,:].imag
                            )))
                        else:
                            beam_imag = 0
                        beam_array[v,p,:,:] = beam_real + 1j*beam_imag
        
        queue.put([rank, beam_array])
        
        
        
    def _extend_beam_in_phi(self, uvb, extend_length=3):
        '''
            This function is credit to pyuvbeam
        '''
        phi = np.rad2deg(uvb.axis1_array)
        beam_array = uvb.data_array

        phi_diff = np.diff(phi)[0]
        phi_length = np.abs(phi[0] - phi[-1]) + phi_diff
        if np.isclose(phi_length, 360, atol=phi_diff):
            # phi wraps around, extend array in each direction to improve interpolation
            phi = np.concatenate(
                (
                    np.flip(phi[:extend_length] * (-1) - phi_diff),
                    phi,
                    phi[-1 * extend_length:] + extend_length * phi_diff,
                )
            )

            low_slice = beam_array[:, :, :, :, :, :extend_length]
            high_slice = beam_array[:, :, :, :, :, -1 * extend_length:]

            beam_array = np.concatenate((high_slice, beam_array, low_slice), axis=5)
            
        return phi, beam_array
