import numpy as np
import os
from pyradiosky import SkyModel
from astropy import units as u
from astropy import constants
from astropy.coordinates import Longitude, Latitude
import healpy as hp
from pathlib import Path
from .data import DATA_PATH

class Sky_Model(object):
    
    def __init__(self, skymodel, freqs, dtype_float=np.float64, nside=None):
        '''
            Initialize the object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                skymodel: Str or Dicionary or List
                    skymodel should be a list of sky sourses ("GLEAM" or "GSM08" or 'GSM16')
                    or dictionary containing keywords for pyradiosky SkyModel object
                    or filename that is a readable format of pyradiosky SkyModel object (e.g., skyh5, fhd, txt, vot)
                    "GLEAM" combines GLEAM I and II along with the peeled sources listed
                    in Table 2 of Hurley-Walker et al. (2017) and Fornax A treated as two point sources.
                    GSM08 and GSM16 indicate model from Oliveira-Costa et. al. (2008) and Zheng et. al. (2016), respectively.
                freqs: Array
                    Frequency (in Hz) at which source flux density will be evaluated.
                dtype_float: Str or Object
                    Data type in creating the map that can be either float32 or float64.
                    This will set corresponding dtype_complex.
                    Default is np.float64.
                nside: Int
                    nside for reading GSM08 or GSM16. Should be specified for GSM.
        '''
        
        
        if not isinstance(skymodel, (list, np.ndarray, tuple)):
            skymodel = [skymodel]
        for model in skymodel:
            if not isinstance(model, (str, dict)):
                raise TypeError('skymodel should a list of string or dictionary')
        
        if not isinstance(freqs, (list, np.ndarray)):
            freqs = [freqs]
        elif isinstance(freqs, tuple):
            raise TypeError('freqs should be either list or numpy array')
            
        freqs = np.asarray(freqs)
        
        if(dtype_float == np.float32):
            dtype_complex = np.complex64
        elif(dtype_float == np.float64):
            dtype_complex = np.complex128
        else:
            raise TypeError('{} should be eighter np.float32 or np.float64')
        
        setattr(self, 'freqs', freqs)
        setattr(self, 'dtype_float', dtype_float)
        setattr(self, 'nside', nside)
        
        freqs_evaluate = freqs * u.Hz
    
        self._set_skymodel(skymodel, freqs_evaluate)
    
    
    
    def _update(self, **kwargs):
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, np.concatenate((getattr(self, k), v), axis=-1))
            else:
                setattr(self, k, v)
                
                
                
    def _construct_gsm_flux_density(self, intensity):
        intensity = np.asarray(list(map(lambda x: hp.ud_grade(x, self.nside), intensity)))

        r = hp.rotator.Rotator(coord='gc')
        intensity = np.asarray(list(map(lambda x: r.rotate_map_alms(x), intensity)))

        pix_area = hp.nside2pixarea(self.nside)
        flux_density = intensity * pix_area

        npix = hp.nside2npix(self.nside)
        pix_idx = np.arange(npix)
        theta_hp, phi_hp = hp.pix2ang(self.nside, pix_idx)

        ra = np.rad2deg(phi_hp)
        dec = np.rad2deg(-theta_hp+np.pi*0.5)
        
        return ra, dec, flux_density
                
                
            
    def _set_skymodel(self, skymodel, freqs_evaluate):
        skymodel_dict = {}
        for i, model in enumerate(skymodel):
            if not isinstance(model, dict):
                if(model.lower() == 'gleam'):
                    filename = os.path.join(DATA_PATH, 'gleam.skyh5')
                    skymodel_obj = SkyModel()
                    skymodel_obj.read(filename)
                    skymodel_obj.at_frequencies(freqs_evaluate)
                    skymodel_dict = {'component': np.repeat('gleam', skymodel_obj.Ncomponents),
                                     'ra': skymodel_obj.ra.degree,
                                     'dec': skymodel_obj.dec.degree,
                                     'stokes_I': skymodel_obj.stokes[0].value}

                elif(model.lower() == 'gsm08'):
                    try:
                        from pygdsm import GlobalSkyModel
                    except:
                        raise ImportError("pygdsm should be installed to import GlobalSkyModel.")

                    if(self.nside is None):
                        raise KeyError('nside should be specified.')

                    gsm_high = GlobalSkyModel(freq_unit='Hz', basemap='haslam')
                    brightness_temp = np.array(gsm_high.generate(freqs_evaluate), dtype=self.dtype_float) # in Kelvin

                    nside_in = int(np.sqrt(brightness_temp.shape[1]/12))
                    K2Jy_sr = 2*constants.k_B*(freqs_evaluate)**2/constants.c**2*1e26
                    intensity = (brightness_temp * K2Jy_sr[:, np.newaxis]).value

                    ra, dec, flux_density = self._construct_gsm_flux_density(intensity)
                    npix = hp.nside2npix(self.nside)

                    skymodel_dict = {'component': np.repeat('gsm08', npix),
                                     'ra': ra,
                                     'dec': dec,
                                     'stokes_I': flux_density}

                elif(model.lower() == 'gsm16'):
                    try:
                        from pygdsm import GlobalSkyModel16
                    except:
                        try:
                            from pygdsm import GlobalSkyModel2016
                        except:
                            raise ImportError("pygdsm should be installed to import GlobalSkyModel2016.")

                    if(self.nside is None):
                        raise KeyError('nside should be specified.')

                    gsm = GlobalSkyModel2016(freq_unit='Hz', data_unit='MJysr', resolution='hi')
                    intensity = np.array(gsm.generate(freqs_evaluate)*1e6, dtype=self.dtype_float)

                    ra, dec, flux_density = self._construct_gsm_flux_density(intensity)
                    npix = hp.nside2npix(self.nside)

                    skymodel_dict = {'component': np.repeat('gsm16', npix),
                                     'ra': ra,
                                     'dec': dec,
                                     'stokes_I': flux_density}

                elif(Path(model).is_file()):
                    skymodel_obj = SkyModel()
                    skymodel_obj.read(model)
                    skymodel_obj.at_frequencies(freqs_evaluate)

                    skymodel_dict = {'component': np.repeat('catalog{}'.format(i),
                                                            skymodel_obj.Ncomponents),
                                     'ra': skymodel_obj.ra.degree,
                                     'dec': skymodel_obj.dec.degree,
                                     'stokes_I': skymodel_obj.stokes[0].value}
            else:
                skymodel_obj = SkyModel(**model)
                skymodel_obj.at_frequencies(freqs_evaluate)

                skymodel_dict = {'component': np.repeat('catalog{}'.format(i),
                                                        skymodel_obj.Ncomponents),
                                 'ra': skymodel_obj.ra.degree,
                                 'dec': skymodel_obj.dec.degree,
                                 'stokes_I': skymodel_obj.stokes[0].value}
                
                
            self._update(**skymodel_dict)
            
    
    
    def update_component_name(self, given_name, new_name):
        '''
            Replace the component name with the use-specified one
            
            Parameters:
            ----------------------------------------------------------------------------------------
                given_name: Str
                    The given name in the .component key
                new_name: Str
                    The new name that will replace for the given name
        '''
        
        idx_match = np.where(self.component == given_name)[0]
        self.component[idx_match] = new_name
        
        
        
    def set_flux_cut(self, component, freq_ref, flux_density_limit):
        '''
            Select sources above certain flux_density_limit in Jy corresponding "component".
            
            Parameters:
            ----------------------------------------------------------------------------------------
                component: Str
                    Sky component that the flux_limit is applied to.
                freq_ref: Float
                    Reference frequency in Hz at which the flux_density cut is evaluated.
                flux_density_limit: Float
                    Flux density limit to cut the sources below the limit.
        '''
        
        idx_match = np.where(self.component == component)[0]
        idx_freq = np.where(np.isclose(self.freqs, freq_ref, atol=1, rtol=0))[0]
        if(len(idx_freq) == 0):
            raise ValueError("freq_ref should be one of the values in sky_model.freqs")
            
        idx_delete = np.where(self.stokes_I[idx_freq[0], idx_match] < flux_density_limit)[0]
        self.component = np.delete(self.component, idx_match[idx_delete])
        self.ra = np.delete(self.ra, idx_match[idx_delete])
        self.dec = np.delete(self.dec, idx_match[idx_delete])
        self.stokes_I = np.delete(self.stokes_I, idx_match[idx_delete], axis=1)