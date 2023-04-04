VisPB
=====

`VisPB` is a Python-based radio interferometric measurement (visibility) simulator. It is specialized to implement fast visibility calculation under per-antenna perturbations. The calculation is parallelized with `multiprocessing`.



Installation
----------

The installation is based on `pip install`. You can simply install it by

`pip install git+https://github.com/vispb/vispb`

Or, clone the directory:

`git clone https://github.com/vispb/vispb`
       
And run `pip install .` in the cloned directory.

All dependencies you need to install are:

* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/install.html)
* [astropy](http://www.astropy.org/)
* [healpy](http://healpy.readthedocs.org/en/latest/)
* [pyuvdata](https://pyuvdata.readthedocs.io/en/latest/uvdata.html)
* [pyradiosky](https://pyradiosky.readthedocs.io/en/latest/)
* [numexpr](https://numexpr.readthedocs.io/en/latest/)
* [tqdm](https://tqdm.github.io/)



Examples
---------

Some tutorials are provided to show how to run the simulation code:

* [single beam simulation](http://nbviewer.ipython.org/github/vispb/vispb/blob/main/tutorial/simulation_50point_sources_single_beam.ipynb)

* [multiple beam simulation](http://nbviewer.ipython.org/github/vispb/vispb/blob/main/tutorial/simulation_50point_sources_multiple_perturbed_beams.ipynb)

* [GLEAM & GSM simulation](http://nbviewer.ipython.org/github/vispb/vispb/blob/main/tutorial/simulation_gleam_gsm_dipole_beam_HERA_configuration.ipynb)