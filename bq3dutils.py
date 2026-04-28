import os
import numpy as np
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import tarfile

# external
import astropy
from astropy import units as u 
from astropy.io import fits
from astropy import constants
from astropy.wcs import WCS

from tqdm.auto import tqdm

# q3dfit
from q3dfit.q3dpro import OneLineData, LineData
from q3dfit import q3dutil
from bq3dfit import bq3dplots
from q3dfit.q3dout import load_q3dout
from q3dfit.q3df import q3dfit
from q3dfit.q3din import q3din

c_kms = constants.c.to('km/s')


# ======================================
# === Compact ===
def compact_fit(filename_q3di, rootdir_new=None):


    assert isinstance(filename_q3di, str)
    q3di = q3dutil.get_q3dio(filename_q3di)

    dirname = os.path.dirname(q3di.outdir).rstrip("/")
    filename_targz = f'{dirname.rstrip("/")}.tar.gz'

    if rootdir_new is not None:

        rootdir_old = os.path.dirname(q3di.infile) + ''

        rootdir_new = rootdir_new.rstrip("/")
        rootdir_old = rootdir_old.rstrip("/")



        #q3di = q3dutil.get_q3dio(filename_q3di)
        q3di.outdir = q3di.outdir.replace(rootdir_old, rootdir_new)
        q3di.infile = q3di.infile.replace(rootdir_old, rootdir_new)
        q3di.logfile = q3di.logfile.replace(rootdir_old, rootdir_new)

        np.save(filename_q3di, q3di)

    compact_directory_targz(dirname, filename_targz)

    # ================
    # Reverse
    #q3di = q3dutil.get_q3dio(filename_q3di)
    q3di.outdir = q3di.outdir.replace(rootdir_new, rootdir_old)
    q3di.infile = q3di.infile.replace(rootdir_new, rootdir_old)
    q3di.logfile = q3di.logfile.replace(rootdir_new, rootdir_old)

    np.save(filename_q3di, q3di)

def compact_directory_targz(dirname, filename_targz=None):
    if filename_targz is None:
        filename_targz = f'{dirname.rstrip("/")}.tar.gz'

    with tarfile.open(filename_targz, "w:gz") as tar:
        tar.add(dirname, arcname=os.path.basename(dirname))

# ======================================
# --------- Q3D ------------------------
def read_range(cube, redshift, cfg_fit, section, key_range, key_range_rest):
    wrange = eval_quantity(cfg_fit.get(section, key_range))
    if wrange is None:
        wrange_rest = eval_quantity(cfg_fit.get(section, key_range_rest))
        if wrange_rest is not None:
            assert isinstance(wrange_rest, u.Quantity)
            wrange = (wrange_rest * (1 + redshift)).to('um')
            wrange = wrange.to(cube.waveunit_out)#.value
        else:
            wrange = None
    return wrange

def read_init(cfg_fit, section, key, unit='km/s'):
    value_init = json.loads(cfg_fit.get(section, key))
    if value_init is not None:
        if is_iterable(value_init):
            value_init = np.array(value_init) * u.Unit(unit)
        else:
            value_init = np.array([value_init]) * u.Unit(unit)
        return value_init
    else:
        return value_init
# =============================================
def spectral_to(q3di, spectral, rest=False, rest_line_name=None, redshift=None,):

    # ====================================
    # Spectral
    xlabel = "Wavelength ($\mu m$)"

    spectral1 = spectral * 1

    if rest:
        if redshift is None:
            redshift = q3di.zsys_gas
        spectral1 = spectral1 / (1 + redshift) 

    if rest_line_name is not None:
        linelist = q3dutil.get_linelist(q3di)

        assert isinstance(rest_line_name, str)
        try:
            rest_line = linelist[linelist['name'] == rest_line_name]['lines']
            spectral1 = (spectral1 / rest_line - 1) * c_kms
        except:
            raise Exception(f"Line not fitted: {rest_line_name}")


    return spectral1

#==============================
def get_continuum_img(q3di_continuum, flux_density=True, use_same_width=True,
                      wavelenth_range=None, filename_noise=None, SNR_cut=3):
 

    # load data
    q3di_continuum = q3dutil.get_q3dio(q3di_continuum)

    # fitted cube
    cube_fitted_cube = q3di_continuum.load_cube()

    # wavelength axis
    unit_spectral = u.Unit(cube_fitted_cube.header_dat['CUNIT3'])
    wave = cube_fitted_cube.wave * unit_spectral

    # cdelt
    print('WARNING: Assuming uniformily spaced spectral axis')
    dw = np.diff(wave)[0].to(u.AA)

    # Load conta data fit
    cube_cont = ContData(q3di_continuum).all_mod.T# Transpose

    unit_spectral = u.Unit(cube_fitted_cube.header_dat['bunit'])
    cube_cont = cube_cont * q3di_continuum.argsreadcube['fluxnorm']
    cube_cont = cube_cont * u.Unit('erg / (cm2 s um)')

    # Get number of good cont data
    cont_bad = (cube_cont == 0) + (np.isnan(cube_cont)) 

    # Mask non fitted spaxels
    cont_bad2D = cont_bad.all(axis=0)
    cont_bad = np.ma.array(cont_bad, mask=np.tile(cont_bad2D, (len(cube_cont), 1, 1)))
    cube_cont = np.ma.array(cube_cont, mask=cont_bad)

    # cube
    if wavelenth_range is not None:
        mask_wave = (wave > wavelenth_range[0]) & (wave < wavelenth_range[1])

        wave = wave[~mask_wave]
        cube_cont = cube_cont[~mask_wave]

    # Cube the continuum fit the same width range
    if use_same_width:
        mask_wave = cont_bad.any(axis=(1, 2))

        wave = wave[~mask_wave]
        cube_cont = cube_cont[~mask_wave]

        bandwidth = wave[-1] - wave[0]
    else:
        raise Exception("Not implemented: Using different bandwidth")
    
        # bandwidth3D = ...

    # Total flux continuum
    # 'WARNING: Assuming uniformily spaced spectral axis'
    #flux_cont = np.sum(cube_cont, axis=0)
    flux_cont = (np.sum(cube_cont.data*dw, axis=0)).to('erg / (cm2 s)')
    flux_cont = np.ma.array(flux_cont, mask=cube_cont.mask.all(axis=0))

    if filename_noise is not None:

        noise = fits.getdata(filename_noise)  * u.Unit('erg / (cm2 s um)')
        uncertainty_flux_cont =  np.sqrt(float(len(wave))) * (noise * dw).to('erg / (cm2 s)')

        mask_snr = (flux_cont.data / uncertainty_flux_cont) < SNR_cut

        flux_cont.mask += mask_snr

    # flux densityy
    if flux_density:
        
        flux_cont_density = flux_cont / bandwidth

        return flux_cont_density

    else:
        return flux_cont


def get_idx_spectral_range(spectral, spectral_range):

    spectral_range = np.array(spectral_range)

    if spectral_range.ndim == 1:
        spectral_range = np.array([spectral_range])
    
    idx = np.zeros(len(spectral), dtype=bool)
    for spectral_range_i in spectral_range:
        assert(len(spectral_range_i) == 2, f"range != 2: {spectral_range_i}")

        idx += ((spectral > spectral_range_i[0]) & (spectral < spectral_range_i[1]))

    return idx

def get_noise(q3di_contsub, cubefile=None,
              cont_restwl_ranges=None, redshift=None, cube=None,
              outfile=None, overwrite=False, threshold_sigma=5):

    # Load q3di data
    if not isinstance(q3di_contsub, q3din):
        q3di_contsub = q3dutil.get_q3dio(q3di_contsub)

    infile0 = q3di_contsub.infile + '' 
    if cubefile is not None:
        q3di_contsub.infile = cubefile

    if cube is None:
        cube = q3di_contsub.load_cube()
        q3di_contsub.infile = infile0

    if redshift is None:
        redshift = q3di_contsub.zsys_gas

    spectral_unit = u.Unit(cube.header_dat['CUNIT3'])
    wl = cube.wave * spectral_unit
    restwl = cube.wave / (redshift + 1)

    if cont_restwl_ranges is None:
        cont_restwl_ranges = np.array([#[0.46037244, 0.47800815],
                                       [0.50993654, 0.52552785]
                                       ]) * u.um
    else:
        cont_restwl_ranges = np.array(cont_restwl_ranges)


    # Define the wavelength ranges for continuum
    idx_cont = get_idx_spectral_range(restwl, cont_restwl_ranges)

    # masked cube
    mask = cube.dq.astype(bool)
    mask += np.isnan(cube.dat) + (cube.dat == 0)
    masked_cube = np.ma.array(cube.dat, mask=mask)
    masked_contcube = masked_cube[:,:,idx_cont]

    # Calculate the standard deviation of the continuum
    cont_std2D = np.std(masked_contcube, axis=2)

    # Mask out regions with signal greater than 5 sigma
    mask_5sig = (np.abs(masked_contcube) / cont_std2D[:,:,np.newaxis]) > threshold_sigma
    masked_contcube.mask = masked_contcube.mask + mask_5sig

    # Recalculate the standard deviation of the masked continuum
    cont_std2D = np.std(masked_contcube, axis=2) * cube.fluxnorm
    cont_std2D = cont_std2D.T

    # Write
    if outfile is not None:
        
        cont_std2D.data[cont_std2D.mask] = np.nan
        cont_std2D = cont_std2D.data

        header = WCS(cube.header_dat).celestial.to_header()

        write_simplefits(cont_std2D, outfile, header=header,
                                overwrite=overwrite)

    return cont_std2D

# --------- Q3D - maps ------------------------
#from q3dfit.q3dpro import OneLineData, ContData, LineData

def load_linemaps(q3di, lineselect='[OIII]5007', cube=None, transpose=True,
                  stats=True, type_mask2D='any', sigmin=None,
                  redchisqmax=None,
                  apply_mask=True, SNR_cut=3, redshift=None, 
                  which_snr_selection='pkflux',
                  which_pkflux_noise='fitsfile',
                  fluxpk_noise_fitsfile=None,
                  which_flux_noise='lmfit',
                  #flux_unit_out='erg/s/cm2', 
                  #bunit_out=None,
                  #spectral_unit_out='um'
                  ):
    """
    Load line maps from a data cube and apply various processing steps.
    Parameters
    ----------
    q3di : object
        Input data object containing the data cube and associated metadata.
    lineselect : str, optional
        The emission line to select, by default '[OIII]5007'.
    cube : object, optional
        Data cube object, by default None.
    transpose : bool, optional
        Whether to transpose the output data, by default False.
    apply_mask : bool, optional
        Whether to apply a mask to the data, by default True.
    SNR_cut : float, optional
        Signal-to-noise ratio cut-off value, by default None.
    redshift : float, optional
        Redshift value to use, by default None.
    which_snr_selection : str, optional
        Method for SNR selection, by default 'flux and pkflux', which is the 
        less restrictive method.
        - 'pkflux',         mask if SNR(pkflux) > snr_CUT
        - 'flux',           mask if SNR(flux) > snr_CUT
        - 'flux or pkflux', mask if (SNR(pkflux) > snr_CUT) OR
                                    (SNR(flux) > snr_CUT
        - 'flux or pkflux', mask if (SNR(pkflux) > snr_CUT) AND
                                    (SNR(flux) > snr_CUT)
    which_fluxpk_amp : str, optional
        Method for peak amplitude selection, by default 'convolved'.
        - 'convolved': result from the fit, where the gaussian profiled was
        convolved with the instrumental dispersion
        - 'observed': 
    which_fluxpk_noise : str, optional
        Method for peak noise selection, by default 'continuum'.
        - 'fitsfile': from a saved file (<fluxpk_noise_fitsfile>)
        - 'lmfit': result from fit
        - 'fluxdensity_uncertainty'
    fluxpk_noise_fitsfile: str, optional
        - fits file with a previouly measured flux density noise map 
    which_flux_noise : str, optional
        Method for flux noise selection, by default 'lmfit'.
        - 'lmfit': result from fit
    stats : bool, optional
        Whether to compute statistical quantities, by default False.
    type_mask2D : str, optional
        Type of 2D mask to apply, by default 'any'.
        - "any": snr_any_below
        - "all: snr_all_below
        - "manxcomp": ncomp_above_snr_equal_maxncomp

    sigmin : float, optional
        Minimum value for the velocity dispersion, by default None.
    redchisqmax : float, optional
        Maximum value for reduced chi-square, by default None.
    Returns
    -------
    linedata : object
        Object containing the processed line data with various attributes.
    Notes
    -----
    The function performs the following steps:
    - Loads line data and computes errors.
    - Un-normalizes and adds units to the data.
    - Computes radial velocity and its error.
    - Computes signal-to-noise ratio (SNR) for different methods.
    - Applies masks based on various criteria.
    - Computes statistical quantities if requested.
    - Optionally transposes the data.

    # Documentation initialized with Copilot
    """

    c = constants.c.to('km/s')

    linedat = LineData(q3di)
    linedata = OneLineData(linedat, lineselect)

    # get errors
    def get_errors(self, linedata):

        self.fluxerr = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.pkfluxerr = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.sigerr = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad
        self.waveerr = \
            np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                     dtype=float) + linedata.bad

        for i in range(0, linedata.maxncomp):
            self.fluxerr[:, :, i] = \
                (linedata.get_flux(lineselect, FLUXSEL='fc'+str(i+1)))['fluxerr']
            self.pkfluxerr[:, :, i] = \
                (linedata.get_flux(lineselect,
                                   FLUXSEL='fc'+str(i+1)+'pk'))['fluxerr']
            self.sigerr[:, :, i] = \
                (linedata.get_sigma(lineselect, COMPSEL=i+1))['sigerr']
            self.waveerr[:, :, i] = \
                (linedata.get_wave(lineselect, COMPSEL=i+1))['waverr']

        return self

    linedata = get_errors(self=linedata, linedata=linedat)

    # laod cube
    if cube is None:
        cube = q3di.load_cube()

    # un-normalize and add units
    u.kms = u.Unit('km/s')

    # flux density unit
    if 'micron' in cube.fluxunit_out:
        bunit = u.Unit(cube.fluxunit_out.replace('micron', 'um'))
    else:
        bunit = u.Unit(cube.fluxunit_out)    
    # spectral unit
    if cube.waveunit_out == 'micron':
        sunit = u.um
    else:
        sunit = u.Unit(cube.waveunit_out)
    # flux unit
    funit = bunit * sunit

    linedata.funit = funit
    linedata.bunit = bunit
    linedata.sunit = sunit

    # Amplitude
    linedata.pkflux = linedata.pkflux * cube.fluxnorm * bunit
    linedata.pkfluxerr = linedata.pkfluxerr * cube.fluxnorm * bunit

    # Velocity Dispersion
    linedata.sig = linedata.sig * u.kms
    linedata.sigerr = linedata.sigerr * u.kms

    # Central wavelength of the fitted Gaussian
    linedata.wave = linedata.wave * sunit
    linedata.waveerr = linedata.waveerr * sunit

    # Flux
    linedata.flux = linedata.flux * cube.fluxnorm * funit
    linedata.fluxerr = linedata.fluxerr * cube.fluxnorm * funit

    # Radial Velocity (in km/s)
    linelist = q3dutil.get_linelist(q3di)
    rest_line = linelist[linelist['name'] == lineselect]['lines'][0] * u.um
    linedata.rest_line = rest_line

    if redshift is None:
        redshift = q3di.zsys_gas

    linedata.vel = ((linedata.wave / (1 + redshift) / rest_line - 1) * c).to(u.kms)
    linedata.velerr = ((linedata.waveerr / (1 + redshift) / rest_line - 1) * c).to(u.kms)
    
    # ======================
    # SNR (Signal-to-Noise Ratio)
    # (SNR) peak flux ==
    # Determine the method for calculating the peak flux error for SNR
    if which_pkflux_noise == 'lmfit':
        linedata.pkfluxerr_for_snr = linedata.pkfluxerr

    elif which_pkflux_noise == 'fluxdensity_uncertainty':
        # Calculate the 2D peak flux error from the cube error
        pkfluxerr_for_snr2D = np.nanmean(cube.err, axis=2) * cube.fluxnorm * bunit
        # Tile the 2D error to match the 3D shape of the flux data
        linedata.pkfluxerr_for_snr = np.tile(pkfluxerr_for_snr2D.T, 
                                             (linedata.flux.shape[2], 1, 1)).T

    elif which_pkflux_noise == 'fitsfile':
        assert fluxpk_noise_fitsfile is not None

        pkfluxerr_for_snr2D = fits.getdata(fluxpk_noise_fitsfile).T * bunit# * cube.fluxnorm * bunit

        # Tile the 2D error to match the 3D shape of the flux data
        linedata.pkfluxerr_for_snr = np.tile(pkfluxerr_for_snr2D.T, 
                                             (linedata.flux.shape[2], 1, 1)).T
    else:
        raise Exception(f'<which_pkflux_noise> not recognized: {which_pkflux_noise}')

    # Calculate the peak flux SNR
    linedata.snr_pkflux = linedata.pkflux / linedata.pkfluxerr_for_snr

    # == (SNR) peak flux ==
    # Determine the method for calculating the flux noise
    if which_flux_noise == 'lmfit':
        linedata.pkfluxerr_for_snr = linedata.fluxerr

    else:
        raise Exception(f'<which_flux_noise> not recognized: {which_flux_noise}')

    # Calculate the flux SNR
    linedata.snr_flux = linedata.flux / linedata.pkfluxerr_for_snr

    # List of keys to be masked
    keylist = ['flux', 'fluxerr', 'sig', 'sigerr', 'wave', 'waveerr',
               'vel', 'velerr', 'pkflux', 'pkfluxerr', 'snr_pkflux', 'snr_flux',
               'pkfluxerr_for_snr']
    
    # ======================
    # stats
    if stats:
        linedata.redchisq = linedat.data['redchisq'] * u.Unit(1)
        linedata.dof = linedat.data['dof'] * u.Unit(1)
        linedata.ndata = linedat.data['ndata'] * u.Unit(1)
        linedata.maxncomp = linedat.data['maxncomp'] * u.Unit(1)

        if 1:
            # Get quantities not saved in the npz file
            # number of free parameters (from lmfit)
            linedata.nparsfree = linedata.ndata - linedata.dof
            # chi square
            linedata.chisq = linedata.redchisq * linedata.dof

            # negatve log(likelihood)
            if 1:
                # Using maximum likelihood from lmfit (probably worng
                _neg2_log_likel = linedata.ndata * np.log(linedata.chisq / linedata.ndata)
                linedata.aic0 = _neg2_log_likel + 2 * linedata.nparsfree
                linedata.bic0 = _neg2_log_likel + np.log(linedata.ndata) * linedata.nparsfree
                linedata.logLmax0 = - 0.5 * _neg2_log_likel

            if 1:

                def get_logLmax(chisq):

                    #logLmax = -0.5 * np.sum(np.log(2 * np.pi * (err**2)) + ((data - model)/err)**2)
                    #logLmax = -0.5 * (np.sum(np.log(2 * np.pi * (err**2))) + chisq
                    logLmax = -0.5 * chisq 

                    return logLmax

                def get_aic(log_likelihood, n_samples, n_params):

                    aic = (2.0 * (n_params - log_likelihood) +
                                    2.0 * n_params * (n_params + 1.0) /
                                    (n_samples - n_params - 1.0))
                    return aic
                    
                def get_bic(log_likelihood, n_samples, n_params):
                    bic = n_params*np.log(n_samples) - 2.0*log_likelihood
                    return bic

                linedata.logLmax = get_logLmax(linedata.chisq)
                linedata.aic = get_aic(linedata.logLmax, 
                                       n_samples=linedata.ndata, 
                                       n_params=linedata.nparsfree)
                linedata.bic = get_bic(linedata.logLmax, 
                                       n_samples=linedata.ndata, 
                                       n_params=linedata.nparsfree)

        keylist += ['redchisq', 'dof', 'ndata', 'maxncomp',
                    'chisq', 'nparsfree', 'logLmax0', 'aic0', 'bic0',  'logLmax', 'aic', 'bic']

    # ======================
    linedata.masked_pars = keylist

    # ======================
    if apply_mask:

        # mask sig, pkflux = 0
        mask_sig0 = linedata.sig == 0
        mask_pkflux0 = linedata.pkflux == 0

        mask_bad = mask_sig0 + mask_pkflux0

        if sigmin is not None:
            mask_sigmin = linedata.sig < sigmin
            mask_bad = (mask_bad.T + mask_sigmin.T).T

        if redchisqmax is not None:
            mask_redchi_sq = linedata.redchisq > redchisqmax
            mask_bad = (mask_bad.T + mask_redchi_sq.T).T

        if SNR_cut is not None:

            mask_snr_flux = (linedata.snr_flux < SNR_cut) +\
                                (~np.isfinite(linedata.snr_flux))

            mask_snr_pkflux = (linedata.snr_pkflux < SNR_cut) +\
                                (~np.isfinite(linedata.snr_pkflux))


            # Determine the method for selecting the SNR
            if which_snr_selection == 'flux or pkflux':
                #linedata.snr = linedata.snr_pkflux | linedata.snr_flux
                mask_snr = mask_snr_flux | mask_snr_pkflux

            elif which_snr_selection == 'flux and pkflux':
                #linedata.snr = linedata.snr_pkflux & linedata.snr_flux
                mask_snr = mask_snr_flux & mask_snr_pkflux

            elif which_snr_selection == 'flux':
                #linedata.snr = linedata.snr_flux
                #mask_snr_flux = (linedata.snr_flux < SNR_cut) +\
                #                  (~np.isfinite(linedata.snr))
                mask_snr = mask_snr_flux

            elif which_snr_selection == 'pkflux':
                #linedata.snr = linedata.snr_pkflux
                #mask_snr_pkflux = (linedata.snr_pkflux < SNR_cut) +\
                #                  (~np.isfinite(linedata.snr))
                mask_snr = mask_snr_pkflux

            else:
                raise Exception(f'<which_snr_selection> not recognized: {which_snr_selection}')

            #mask_SNR = (linedata.snr < SNR_cut) + (~np.isfinite(linedata.snr)) 
            mask = mask_bad + mask_snr
        else:
            mask = mask_bad

        if cube is None:
            cube = q3di.load_cube()

        # mask DQ
        mask2D_dq = cube.dq.astype(bool).all(axis=-1)
        mask3D_dq = np.tile(mask2D_dq.T, (linedata.flux.shape[-1], 1, 1)).T

        # mask_nan
        mask_nan = mask3D_dq

        # Masked array
        mask3D = mask + mask_nan

        # (in a given spaxel) Mask all components in a any has SNR < SNR_cut?
        if type_mask2D == "any":
            mask2D = mask3D.any(axis=2)
        elif type_mask2D == "all":
            mask2D = mask3D.all(axis=2)
        elif type_mask2D == "manxcomp":
            mask2D = np.sum(~mask3D, axis=2) == 1
        else:
            raise Exception(f'<type_mask2D> not rexognized: {type_mask2D}')

        mask3D = (mask3D.T + mask2D.T).T

        for key in keylist:
            data = getattr(linedata, key)

            if data.ndim == 3:
                data_masked = np.ma.array(data, mask=mask3D)
            elif data.ndim == 2:
                data_masked = np.ma.array(data, mask=mask2D)
            else:
                raise Exception(f"<ndim> = {ndim} not implemented.")

            setattr(linedata, key, data_masked)

        linedata.mask3D = mask3D
        linedata.mask2D = mask2D
        keylist += ['mask2D', 'mask3D']

    # Transpose
    if transpose:
        for key in keylist:
            setattr(linedata, key, getattr(linedata, key).T)

    #linedata.linedat = linedat
    return linedata


class ContData:
    def __init__(self, q3di):
        filename = q3di.label+'.cont.npy'
        datafile = os.path.join(q3di.outdir,filename)
        if os.path.exists(datafile) != True:
            print('ERROR: continuum ('+filename+') file does not exist')
            return None


        self.data = self.read_npy(datafile)

        for key in list(self.colname):
            setattr(self, key, self.data[key])

        return

    def read_npy(self, datafile):
        dataout = np.load(datafile, allow_pickle=True).item()
        self.colname = dataout.keys()
        return dataout

def LineData_get_errors(self, linedata, lineselect):

    self.wave = \
        np.zeros((linedata.ncols, linedata.nrows, linedata.maxncomp),
                 dtype=float) + linedata.bad

    # cycle through components to get maps
    for i in range(0, linedata.maxncomp):
        self.flux[:, :, i] = \
            (linedata.get_flux(lineselect, FLUXSEL='fc'+str(i+1)))['flux']
        self.pkflux[:, :, i] = \
            (linedata.get_flux(lineselect,
                               FLUXSEL='fc'+str(i+1)+'pk'))['flux']
        self.sig[:, :, i] = \
            (linedata.get_sigma(lineselect, COMPSEL=i+1))['sig']
        self.wave[:, :, i] = \
            (linedata.get_wave(lineselect, COMPSEL=i+1))['wav']
    # No. of components on a spaxel-by-spaxel basis
    self.ncomp = linedata.get_ncomp(lineselect)

    return 


#===========================
# ======== ctools =========
def clean():

    plt.close('all')
    import gc
    gc.collect()

def smooth(y, sigma=1, sfilter=None, convolve_opts={}, **filterargs):

    if sfilter is None:
        if y.ndim == 1:
            sfilter = 'gauss1d'

        elif y.ndim == 2:
            sfilter = 'gauss2d'

        else:
            raise Exception("<sfilter> not implemented:",  sfilter)

    if sfilter == 'gauss1d':
        #stddev
        from astropy.convolution import Gaussian1DKernel, convolve
        smoothed = convolve(y, Gaussian1DKernel(sigma, **filterargs),
                            **convolve_opts)
    elif sfilter == 'gauss2d':
        #stddev
        from astropy.convolution import Gaussian2DKernel, convolve
        if 'x_stddev' not in filterargs:
            filterargs['x_stddev'] = sigma
        smoothed = convolve(y, Gaussian2DKernel(**filterargs),
                            **convolve_opts)
    elif sfilter == 'ndimage':
        #stddev
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(y, sigma, **filterargs)
        
    elif sfilter == 'savgol':
        pass

    return smoothed

def write_multifits(data, fitsfile, header, name, overwrite=False):

    data1 = deepcopy(data)
    if not is_iterable(data):
        data1 = [data1]
    if not is_iterable(name):
        name1 = [name]
    else:
        name1 = name
    if not is_iterable(header):
        header1 = [header]
        if len(header1) == 1:
            header1 = header1 * len(data)
    else:
        header1 = header

    hdu_pri = fits.PrimaryHDU()
    hdu_data = []
    for i in range(len(data1)):
        if isinstance(data[i], u.quantity.Quantity):
            header2 = deepcopy(header1[i])
            header2['bunit'] = data[i].unit.to_string()
            data[i] = data[i].to_value()
        else:
            header2 = header1[i]

        if data[i].dtype == bool:
            data[i] = data[i].astype(int)

        hdu_data += [fits.ImageHDU(data=data[i], header=header2, name=name[i])]

    hdu_list = fits.HDUList([hdu_pri] + hdu_data)
    hdu_list.writeto(fitsfile, overwrite=overwrite)

def write_simplefits(data, fitsfile, header, overwrite=False):
    #assert fitsfile is None, "Inform <fitsfile>

    header1 = deepcopy(header)
    data1 = deepcopy(data)

    if isinstance(data, u.quantity.Quantity):
        header1['bunit'] = data.unit.to_string()
        data1 = data1.to_value()

    if data1.dtype == bool:
        data1 = data1.astype(int)

    hdu_pri = fits.PrimaryHDU(data=data1, header=header1)
    hdu_pri.writeto(fitsfile, overwrite=overwrite)

# ======================================
def subtract_continuum(q3di, out_cubefile=None, overwrite=False):

    if not isinstance(q3di, q3din):
        q3di = q3dutil.get_q3dio(q3di)

    # Load cube
    in_cubefile = q3di.infile
    cube = q3di.load_cube()
    hdu = fits.open(in_cubefile)

    varext = q3di.varext
    datext = q3di.datext
    dqext = q3di.dqext

    # Load contfit
    contdata = ContData(q3di)
    cont = contdata.all_mod.T * q3di.argsreadcube['fluxnorm']

    # subtract continuum
    hdu[datext].data = hdu[datext].data - cont

    # Mask bad values
    mask2D = (cont == 0).all(axis=0)
    hdu[dqext].data[:, mask2D] = 1

    # Mask outside fitrange
    if q3di.fitrange is not None:
        idxfit = (cube.wave > q3di.fitrange[0].value) &\
                 (cube.wave < q3di.fitrange[-1].value)
        hdu[dqext].data[~idxfit] = 1
        hdu[varext].data[~idxfit] = np.nan
        hdu[datext].data[~idxfit] = np.nan
    
    # write to file
    if out_cubefile is None:
        out_cubefile = in_cubefile.replace('.fits',
                          f'_contsub_REGION.fits')

    hdu.writeto(out_cubefile, overwrite=overwrite)


def save_initial_conditions(q3di, fluxpk_noise_fitsfile, SNR_cut=3, line='[OIII]5007',
                            niter_extend=3, smooth_sigma1=3, smooth_sigma2=0.5,
                            smooth_sigma_final=1,
                            plot=True):

    if not isinstance(q3di, q3din):
        
        q3di = q3dutil.get_q3dio(q3di)

    # load cube
    cube = q3di.load_cube()

    # load linemaps
    opts_load_maps = dict(cube=cube, transpose=True, apply_mask=True,
                          SNR_cut=SNR_cut, 
                          fluxpk_noise_fitsfile=fluxpk_noise_fitsfile)

    linemaps = load_linemaps(q3di, line, **opts_load_maps) 

    # z, peak, sig initial values
    zinit = (linemaps.wave.data / linemaps.rest_line - 1).value
    peakinit = linemaps.pkflux.data.value
    siginit = linemaps.sig.data.value

    # mask nan
    mask = linemaps.wave.mask
    zinit[mask] = np.nan
    peakinit[mask] = np.nan
    siginit[mask] = np.nan

    # mask dq
    mask_dq = (cube.dq.T == 1).all(axis=0)

    # Do for every parameter (zinit, peakinit, siginit)
    for parinit in (zinit, peakinit, siginit):

        convolve_opts = dict(boundary='extend')

        # loop over parameters
        for pari in parinit:

            # Smooth (extend region fittef)
            # Every iteration growths the region with initial values data
            for iter in range(niter_extend):
                maskpari = np.isnan(pari)
                pari[maskpari] = smooth(pari, smooth_sigma1, 
                                          convolve_opts=convolve_opts)[maskpari]

            # Smooth again (less agressive)
            pari[maskpari] = smooth(pari, smooth_sigma2,
                                          convolve_opts=convolve_opts)[maskpari]

            # Replace remaining nan values with mean
            maskpar = np.isnan(pari)
            pari[maskpar] = np.nanmean(pari)
            
            pari[mask_dq] = np.nan
        
            # Final smooth
            goodpar = ~np.isnan(pari)
            pari[goodpar] = smooth(pari, smooth_sigma_final,
                          convolve_opts=convolve_opts)[goodpar] 

    if plot:
        fig, ax = plt.subplots(2, 3)
        ax = ax.ravel()

        vpercent = [1, 99] 
        bq3dplots.plot_quantity_map(peakinit[0], axis=ax[0],  doLog=True, 
                                vpercent=vpercent,
                                do_contour=False,
                                do_colorbar=True,
                                cmap='inferno',
                                )
        
        vpercent = [1, 99] 
        bq3dplots.plot_quantity_map(zinit[0], axis=ax[1],  doLog=False, 
                                vpercent=vpercent,
                                do_contour=False,
                                do_colorbar=True,
                                cmap='coolwarm',
                                )

        vpercent = [1, 99] 
        bq3dplots.plot_quantity_map(siginit[0], axis=ax[2],  doLog=False, 
                                vpercent=vpercent,
                                do_contour=False,
                                do_colorbar=True,
                                cmap='inferno',
                                )

        #ax[3].hist(np.log10(peakinit[0]).ravel())
        ax[3].hist(np.log10(peakinit[0]).ravel())
        ax[4].hist(zinit[0].ravel())
        ax[5].hist(siginit[0].ravel())

        ax[0].set_title('peak')
        ax[1].set_title('z')
        ax[2].set_title('sig')

        fig.tight_layout()
        fig.savefig(f'figs/hist_vel_{q3di.label}.png')

    outdir_guess = os.path.dirname(q3di.infile) + '/aux'
    filename_guess = f'{outdir_guess}/init_{q3di.label}.fits'
    if not os.path.exists(outdir_guess):
        os.makedirs(outdir_guess)

    write_multifits([peakinit, zinit, siginit], filename_guess,
                     header=None, name=['peakinit', 'zinit', 'siginit'],
                     overwrite=True)

def save_maps(q3di, q3di_continuum, opts_load_maps={}, save_snr_max=True):

    opts_load_maps1 = dict(transpose=True, apply_mask=True, stats=True,
                            SNR_cut=3, type_mask2D='any',
                            sigmin=10*u.Unit('km/s'),
                            redchisqmax=200,
                            which_snr_selection='pkflux',
                            which_pkflux_noise='fitsfile',
                            fluxpk_noise_fitsfile=None,
                            which_flux_noise='lmfit',
                        )
    opts_load_maps1.update(opts_load_maps)


    linemaps = {}

    key_list = 'pkflux', 'vel', 'sig', 'flux', 'snr_pkflux'


    q3di = q3dutil.get_q3dio(q3di)
    cube = q3di.load_cube()
    opts_load_maps1['cube'] = cube

    linelist = q3dutil.get_linelist(q3di)
    linenames = linelist['name'].value

    img_cont = get_continuum_img(q3di_continuum,
                                        flux_density=True,
                                        use_same_width=True,
                                        wavelenth_range=None)


    data_dict = {}

    for line in linenames:
        linemaps[line] = load_linemaps(q3di, line, **opts_load_maps1) 

        for key in key_list:
            data_dict[f'{line}_{key}'] = getattr(linemaps[line], key)
        
    data_dict['cont_optical'] =  img_cont


    header = fits.getheader(cube.infile, extname='SCI')
    header0 = WCS(header).celestial.to_header()

    ext_names = ['cont_optical']+ list(data_dict.keys())
    ext_data = []
    ext_header = []
    from copy import deepcopy
    for ext in ext_names:
        print(ext)

        datai = data_dict[ext].data.value
        ext_data += [datai] 


        headeri = deepcopy(header0)
        headeri['EXTNAME'] = ext
        headeri['bunit'] = data_dict[ext].data.unit.to_string()
        ext_header += [headeri]

    outdir = os.path.dirname(q3di.infile) + '/outfitmaps/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)            

    sufix_fitmaps =  f"_{q3di.label}_fitmaps.fits"
    filename_fitmaps = f'{outdir}/{os.path.basename(q3di.infile).replace(".fits", sufix_fitmaps)}'

    write_multifits(data=ext_data,
                            fitsfile=filename_fitmaps,
                            name=ext_names,
                            header=ext_header, overwrite=True)

    if save_snr_max:
        sufix_snr =  f"_{q3di.label}_snr.fits"
        filename_snr = f'{outdir}/{os.path.basename(q3di.infile).replace(".fits", sufix_snr)}'
        snr_all = np.array([data_dict[ext].data for ext in ext_names if 'snr' in ext]) 
        SNR = np.max(snr_all, axis=(0,1))
        header_snr= deepcopy(header0)
        header_snr['EXTNAME'] = 'SNR'
        header_snr['bunit'] = 1
        write_simplefits(data=SNR,
                                fitsfile=filename_snr,
                                header=header_snr, overwrite=True)

# ==============================================
def get_filelab(q3di, col, row):

    q3dii = q3dutil.get_q3dio(q3di)

    filelab = '{0.outdir}{0.label}'.format(q3dii)
    filelab += '_{:04d}'.format(col) + '_{:04d}'.format(row) + '.npy'
    
    return filelab

# ======================================
def loop_spaxel_refit(q3di, col, row, onefit=False, quiet=True, 
                      ignore_masked_spaxels=True, nocrash=True, ncores=1,
                      opts_q3dfit={}, do_base_fit=True,
                      refit_max_nfev=False, refit_no_uncertainty=False,
                      argslinefit_iterlist_dict={},
                      cube=None, filename_q3di=None):
    '''
    col, row:
        +1 index

    '''


    opts_q3dfit1 = dict(quiet=quiet, onefit=onefit, 
                        ignore_masked_spaxels=ignore_masked_spaxels,
                        nocrash=nocrash, ncores=ncores)
    opts_q3dfit1.update(opts_q3dfit)


    # ---------- First fit ----------
    if do_base_fit:
        #print(f"[col,row] = [{col:2}, {row:2}] | (base)")#, q3di.argslinefit)

        if not isinstance(q3di, q3din):
            q3di_object = q3dutil.get_q3dio(q3di)
            #print(q3di_object.argslinefit)


        #print('(1) Fitting cube in multi core mode')
        q3dfit(q3di, cols=col, rows=row, **opts_q3dfit1)

    # ---------- max_nfev reached ----------
    if refit_max_nfev:
        #print('(2a) Reffiting spaxels that reached max_nfev')

        argslinefit_iterlist_maxiter = argslinefit_iterlist_dict['maxiter']
        iter_argslinefit(q3di, col, row, opts_q3dfit1,
                         argslinefit_iterlist_maxiter,
                         type_error='maxiter', cube=cube)

        #q3dfit(q3di,cols=cols, rows=rows, **opts_q3dfit1)

    # ---- amplitude = 0 ->> caused by no uncertainty from lmfit -------
    if refit_no_uncertainty:
        #print('(2b) Reffiting spaxels with no uncertainty calculated.')

        argslinefit_iterlist_no_uncertainty = argslinefit_iterlist_dict['no_uncertainty']
        iter_argslinefit(q3di, col, row, opts_q3dfit,
                         argslinefit_iterlist_no_uncertainty,
                         type_error='no_uncertainty', cube=cube)



def loop_spiral(q3di, col_spiral_center, row_spiral_center, 
                cols, rows, opts_q3dfit, spiral_radius_average=1.1, 
                do_base_fit=True,
                refit_max_nfev=False, refit_no_uncertainty=False,
                argslinefit_iterlist_dict={}, cube=None,
                #filename_q3di=None
                ):

    # opts loop spaxel refit
    opts_loop_spaxel_refit = dict(
                         do_base_fit=do_base_fit, refit_max_nfev=refit_max_nfev, 
                         refit_no_uncertainty=refit_no_uncertainty,
                         argslinefit_iterlist_dict=argslinefit_iterlist_dict,
                         cube=cube)

    # filename
    if 0:
        if filename_q3di is not None:
            q3di_run = filename_q3di
        else:
            q3di_run = q3di

    else:
        if isinstance(q3di, q3din):
            q3di_run = q3di
            
            filename_q3di = None
        else:
            filename_q3di = q3di# = filename_q3di
            q3di = q3dutil.get_q3dio(filename_q3di)

            q3di_run = filename_q3di


    # ===========================================
    # Save initial argslinefit
    argslinefit0 = {}
    argslinefit0.update(q3di.argslinefit)

    # ===========================================
    # Get spiral spaxels
    nspax, xcolarr, yrowarr = get_spiral_spaxels(
                                q3di,
                                col_spiral_center, row_spiral_center,
                                cols, rows, cube=cube)

    if nspax == 0:
        print('nspaxel (for loop) = 0')
        return

    # Initialize parameters arrays
    nlines = len(q3di.lines)
    ncomp = q3di.maxncomp
    #npars = nlines * ncomp * 3            

    peakinit_arr = np.zeros((nspax, ncomp, nlines)) * np.nan
    zinit_arr = np.zeros((nspax, ncomp, nlines)) * np.nan
    siginit_arr = np.zeros((nspax, ncomp, nlines)) * np.nan

    # ======= First: Fit central spaxel ==========
    peakinit_gas = deepcopy(q3di.peakinit_gas)
    #q3dfit(q3di,  
    #    #filename_q3di,
    #    cols=colarr[0], rows=rowarr[0], **opts_q3dfit)
    print(f"[col,row] = [{xcolarr[0]+1:2}, {yrowarr[0]+1:2}] | central")
    loop_spaxel_refit(q3di_run, xcolarr[0]+1, yrowarr[0]+1,#xcolarr[0]+1, yrowarr[0]+1, 
                      opts_q3dfit=opts_q3dfit, **opts_loop_spaxel_refit)

    q3di.peakinit_gas = peakinit_gas

    # Read results
    q3do = load_q3dout(q3di, xcolarr[0]+1, yrowarr[0]+1)
    #p_i = np.array([item for key, item in q3do.param.items()\
    #                if 'SPECRES' not in key.upper()])
    rest_line = q3do.linelist['lines'].value

    # Get resulting parameters
    peakinit_gas_i = q3do.line_fitpars['fluxpk'].to_pandas().values
    waveinit_gas_i = q3do.line_fitpars['wave'].to_pandas().values
    siginit_gas_i = q3do.line_fitpars['sigma'].to_pandas().values
    zinit_gas_i = (waveinit_gas_i / rest_line - 1)

    # Save parameters in the arrays (1st element)
    peakinit_arr[0] = peakinit_gas_i
    zinit_arr[0] = zinit_gas_i
    siginit_arr[0] = siginit_gas_i

    # ======= Second: Loop over the remaining spaxels ==========
    ispaxarr = np.arange(1, nspax)
    #ixcolarr = xcolarr[ispaxarr]
    #iyrowarr = yrowarr[ispaxarr]
    #for ispax in ispaxarr:
    for ispax in tqdm(ispaxarr, total=nspax-1, desc='(spiral loop) Spaxel: ',
                      miniters=100, maxinterval=1):
    #for ispax, col, row in tqdm(zip(ispaxarr, ixcolarr, iyrowarr), total=nspax-1, desc=f'(spiral loop) | [col,row] = [{col:2}, {row:2}] | Spaxel: '):
        # row and col of the spaxel
        xcol, yrow = xcolarr[ispax], yrowarr[ispax]
        col, row = xcol + 1, yrow + 1
        #print(f"[col,row] = [{col:2}, {row:2}]")

        # Mask spaxels outside the radius (and nan)
        mask_nan = np.isnan(peakinit_arr)

        radius_xy = np.sqrt((xcolarr-xcol)**2 + (yrowarr-yrow)**2)
        mask_radius = (radius_xy > spiral_radius_average)

        mask = mask_nan + mask_radius[:, np.newaxis, np.newaxis]

        # Average non-masked parameters (inside radius)  
        peakinit_gas_i = np.average(np.ma.array(peakinit_arr, mask=mask), axis=0)
        zinit_gas_i = np.average(np.ma.array(zinit_arr, mask=mask), axis=0)
        siginit_gas_i = np.average(np.ma.array(siginit_arr, mask=mask), axis=0)
        if (peakinit_gas_i == 0).all():
            # For the case of an isolated spaxel, use only nan mask (forget radius mask)
            #print('\n\n********************************************************\n\n')

            if 0:
                print("Removing radius mask for average of initial values")
                mask = mask_nan

                peakinit_gas_i = np.average(np.ma.array(peakinit_arr, mask=mask), axis=0)
                zinit_gas_i = np.average(np.ma.array(zinit_arr, mask=mask), axis=0)
                siginit_gas_i = np.average(np.ma.array(siginit_arr, mask=mask), axis=0)

            if 0:
                print("Isolated spaxel: Using result from first fit as the initial guess.")

                peakinit_gas_i = peakinit_arr[0] * 1
                zinit_gas_i = zinit_arr[0] * 1
                siginit_gas_i = siginit_arr[0] * 1

            if 1:
                print("Selecting the 10 closest good pixels")
                # Mask with good only pixels (fitted ones)
                mask_nan_spaxel = np.isnan(peakinit_arr).any(axis=(1,2))

                # array with radius of good spaxels (sorted)
                radius_xy_good_sorted = np.sort(radius_xy[~mask_nan_spaxel])

                # maximum idx of above array
                #idx_radius_good_max = len(radius_xy_good) - 1
                # Total number of good pixels
                Npix_good_tot = len(radius_xy_good_sorted)
                # If there is the total number of good pixels is smaller then 
                # <Ngood_pix>, update its value to the maximum number
                Ngood_pix = 10
                if Npix_good_tot < Ngood_pix: 
                    Ngood_pix = Npix_good_tot
                idx_Ngood_pix = Ngood_pix - 1

                # new spiral radius maximum
                spiral_radius_average_fix = radius_xy_good_sorted[idx_Ngood_pix]*1.001
                # new nask radius, using the new spiral radius
                radius_xy = np.sqrt((xcolarr-xcol)**2 + (yrowarr-yrow)**2)
                mask_radius = (radius_xy > spiral_radius_average_fix)

                # new mask
                mask = mask_nan + mask_radius[:, np.newaxis, np.newaxis]

                # new initial guesses
                peakinit_gas_i = np.average(np.ma.array(peakinit_arr, mask=mask), axis=0)
                zinit_gas_i = np.average(np.ma.array(zinit_arr, mask=mask), axis=0)
                siginit_gas_i = np.average(np.ma.array(siginit_arr, mask=mask), axis=0)
        #print(f'N mask_radius pixel {(~mask_radius).sum()}')

        # Update guesses for the current spaxel that is being fitted 
        for idx_line, iline in enumerate(q3di.lines):

            q3di.peakinit_gas[iline][col-1, row-1] = peakinit_gas_i[:,idx_line]
            q3di.zinit_gas[iline][col-1, row-1] = zinit_gas_i[:,idx_line]
            q3di.siginit_gas[iline][col-1, row-1] = siginit_gas_i[:,idx_line]

        # Reset argslinefit, to ensure that the loop starts again with the
        # default value, not with the final dict from iter_argslinefit 
        q3di.argslinefit.update(argslinefit0)

        # Save
        if filename_q3di is not None:
            np.save(filename_q3di, q3di)
        else:
            q3di_run = q3di
        

        # Fit spaxel
        # !! using q3di object because initial values are not stored yet
        filename_xy = get_filelab(q3di, col, row)

        try:
        #if 1:
            #loop_spaxel_refit(q3di,
            loop_spaxel_refit(q3di_run,
                              col, row, opts_q3dfit=opts_q3dfit,
                              **opts_loop_spaxel_refit)
            goodfit = True
        except:
        #else:
            goodfit = False

            if os.path.isfile(filename_xy):
                os.remove(filename_xy)


        # Read results
        #try:
        if goodfit:
            
            q3do = load_q3dout(q3di, col, row)
            #p_i = np.array([item for key, item in q3do.param.items()\
            #                if 'SPECRES' not in key.upper()])

            # Get resulting parameters
            peakinit_gas_i = q3do.line_fitpars['fluxpk'].to_pandas().values
            waveinit_gas_i = q3do.line_fitpars['wave'].to_pandas().values
            siginit_gas_i = q3do.line_fitpars['sigma'].to_pandas().values
            zinit_gas_i = (waveinit_gas_i / rest_line - 1)

            # Save parameters in the arrays (for future use in the loop)
            peakinit_arr[ispax] = peakinit_gas_i
            zinit_arr[ispax] = zinit_gas_i
            siginit_arr[ispax] = siginit_gas_i
        #except:
        else:
            print(f"ERROR:  [col,row] = [{col:2}, {row:2}] : ERROR")

            continue

def get_wvel_q3di(q3di, line, spectral=None, rest=False, redshift=0):

    if spectral is None:
        cube = q3di.load_cube()
        spectral = cube.wave

    spectral1 = spectral * 1
    if rest:
        spectral1 = spectral1 / (1 + redshift) 

    linelist = q3dutil.get_linelist(q3di)
    rest_line = linelist[linelist['name'] == line]['lines']
    wvel = (spectral1 / rest_line - 1) * c_kms

    return wvel

# =====================================
def get_bad_spaxels_to_iter(q3di, type_error, 
                            xcolarr=None, yrowarr=None,
                            cols=None, rows=None,
                            cube=None):

    if (xcolarr is None) or (yrowarr is None):
        nspax, xcolarr, yrowarr = q3dutil.get_spaxels(
                                cube, cols=cols, rows=rows, 
                                ignore_masked_spaxels=True)
    else:
        nspax = len(xcolarr)

    q3di1 = q3dutil.get_q3dio(q3di)

    spaxels_to_iter2D = np.zeros((q3di1.ncols, q3di1.nrows), dtype=bool)
    #k = 0
    for xcol, yrow in tqdm(zip(xcolarr, yrowarr), total=nspax,
                               desc='(refit-preselect) Spaxel: '):
    #for xcol, yrow in zip(xcolarr, yrowarr):
        #k = k+1
        #print(k, nspax)
        col, row = xcol + 1, yrow + 1

        if os.path.isfile(get_filelab(q3di1, col, row)):
            #print(k)

            q3do = load_q3dout(q3di1, col, row)

            # Test error
            bool_error = get_bool_error(q3do, type_error)

            spaxels_to_iter2D[xcol, yrow] = bool_error
            #if bool_error:
            #    print(f'(col, row) = {col, row}     |  (y, x) = {yrow, xcol}')
        #else:
        #    print(k, 'NO')

    return spaxels_to_iter2D

def get_bool_error(q3do, type_error):
    if type_error == 'maxiter':
        return (q3do.fitstatus == 0)
    elif type_error == 'no_uncertainty':
        if q3do.line_fit is None:
            return True
        else:
            return (None in np.array([item \
                                     for key, item in q3do.perror.items()]))
    else:
        raise Exception('<type_error> not recognized')

def iter_argslinefit(q3di, cols, rows, opts_q3dfit, 
                     argslinefit_iterlist, 
                     type_error='maxiter', 
                     cube=None, ncores=1):
    '''
    Input
    -----

    q3di: q3din object
    col, row: int
        Col, row to be refitted if needed (+1 idx)
    opts_q3dfit: dict
        Dictionary with arguments to be used in q3dfit
    argslinefit_iterlist: list of dict
        List of argslinefit dictionaries to be used in each iteration
    type_error: str
        'maxiter' or 'no_uncertainty'
    '''



    # Set opts_q3dfit dictionary
    opts_q3dfit1 = {}
    opts_q3dfit1.update(opts_q3dfit)

    # Save
    dq_original_module = deepcopy(cube.dq)

    q3di1 = q3dutil.get_q3dio(q3di)
    argslinefit_original = deepcopy(q3di1.argslinefit)

    # Load cube
    if cube is None:
        cube = q3di1.load_cube()


    # Pre-select spaxels
    #nspax, xcolarr, yrowarr = q3dutil.get_spaxels(
    #                        cube, cols=cols, rows=rows,
    #                        ignore_masked_spaxels=True)
    spaxels_to_iter2D = get_bad_spaxels_to_iter(q3di1, type_error, 
                                                xcolarr=None, yrowarr=None, 
                                                cols=cols, rows=rows,
                                                cube=cube)

    # Update dq
    cube.dq[~spaxels_to_iter2D] = 1
    # save updated dq extension
    #cube.writefits(q3di1.infile)
    hdu = fits.open(cube.infile)
    hdu['DQ'].data = cube.dq.T.astype(int)
    hdu.writeto(cube.infile, overwrite=True)

    # Iterate over spaxels    
    #nspax, xcolarr, yrowarr = q3dutil.get_spaxels(
    #                        cube, cols=cols, rows=rows,
    #                        ignore_masked_spaxels=True)
    nspax, xcolarr, yrowarr = get_spiral_spaxels(q3di1, 
                                                 col_spiral_center=None,
                                                 row_spiral_center=None,
                                                 cols=cols, rows=rows,
                                                 cube=cube)

    #print(f'ncores = {ncores}')
    filename_q3di = None
    if ncores == 1:

        #print('Reffiting spaxels with different tol. values')
        #for col, row in tqdm(zip(colarr, rowarr), total=nspax,
        #                    desc='(refit) Spaxel: '):
        for xcol, yrow in zip(xcolarr, yrowarr):
            col, row = xcol + 1, yrow + 1

            # Update opts_q3dfit
            opts_q3dfit1.update(dict(cols=col, rows=row))

            # Load previous fit result
            #try:
            if os.path.isfile(get_filelab(q3di, col, row)):
                q3do = load_q3dout(q3di, col, row)
            #except:
            else:
                print(f"(col, row) = ({col}, {row}) | No data")#, q3di.argslinefit)
                continue

            # Test error
            bool_error = get_bool_error(q3do, type_error)

            iter_idx = 0
            while bool_error & (iter_idx < len(argslinefit_iterlist)):

                print(f"(col, row) = ({col}, {row}) | iter {iter_idx+1} | ({type_error})")

                # Update argslinefit
                if isinstance(q3di, q3din):

                    q3di.argslinefit.update(argslinefit_iterlist[iter_idx])

                    # handle possible error
                    if 'method' in q3di.argslinefit:
                        if q3di.argslinefit['method'] == 'leastsq':
                            q3di.argslinefit.pop('diff_step', None)

                    #print(q3di.argslinefit)
                else:

                    # q3di is the filename, not the q3din instance
                    # q3di: filename, q3di_object: q3din instance 
                    filename_q3di = q3di  
                    q3di_object = q3dutil.get_q3dio(q3di)
                    q3di_object.argslinefit.update(argslinefit_iterlist[iter_idx])

                    # handle possible error
                    if 'method' in q3di_object.argslinefit:
                        if q3di_object.argslinefit['method'] == 'leastsq':
                            q3di_object.argslinefit.pop('diff_step', None)

                    # save
                    np.save(filename_q3di, q3di_object)

                    #print(q3di_object.argslinefit)


                # Re-fit
                q3dfit(q3di, **opts_q3dfit1)

                # ====== Get info for next iteration ======
                iter_idx += 1

                # Load new fit for bool_error test
                q3do = load_q3dout(q3di, col, row)

                # Update bool_error
                bool_error = get_bool_error(q3do, type_error)

    else:

        #for iter_idx in range(len(argslinefit_iterlist)):
        iter_idx = 0
        nspax_to_iter = np.sum(spaxels_to_iter2D)
        while (nspax_to_iter > 0) & (iter_idx < len(argslinefit_iterlist)):

            # Only works if q3di is a filename
            assert isinstance(q3di, str)
            assert os.path.isfile(q3di)

            # Secure that the fit will the number of cores is not lower than
            # the number of spaxels to fit
            if ncores > nspax_to_iter:
                opts_q3dfit1['ncores'] = nspax_to_iter
            else:
                opts_q3dfit1['ncores'] = ncores

            # q3di is the filename, not the q3din instance
            # q3di: filename, q3di_object: q3din instance 
            filename_q3di = q3di  
            q3di_object = q3dutil.get_q3dio(q3di)
            q3di_object.argslinefit.update(argslinefit_iterlist[iter_idx])

            print(f"   Iter {iter_idx} | nspaxels = {nspax_to_iter}")
            print(f"   argslinefit = {q3di_object.argslinefit}")

            # handle possible error
            if 'method' in q3di_object.argslinefit:
                if q3di_object.argslinefit['method'] == 'leastsq':
                    q3di_object.argslinefit.pop('diff_step', None)

            # save
            np.save(filename_q3di, q3di_object)

            # Re-fit
            q3dfit(filename_q3di, **opts_q3dfit1)

            # -----------------------------------------
            iter_idx += 1

            # Get remaining spaxels to fit and update cube.dq
            spaxels_to_iter2D = get_bad_spaxels_to_iter(q3di_object, type_error, 
                                                xcolarr=None, yrowarr=None, 
                                                cols=cols, rows=rows,
                                                cube=cube)
            # update cube.dq
            cube.dq[~spaxels_to_iter2D] = 1
            # save updated dq extension
            #cube.writefits(q3di_object.infile)
            hdu = fits.open(cube.infile)
            hdu['DQ'].data = cube.dq.T.astype(int)
            hdu.writeto(cube.infile, overwrite=True)

            # new number of spaxels to fit
            nspax_to_iter = np.sum(spaxels_to_iter2D)

    # Reset DQ mask
    cube.dq = dq_original_module
    # save updated dq extension
    #cube.writefits(q3di1.infile)
    hdu = fits.open(cube.infile)
    hdu['DQ'].data = cube.dq.T.astype(int)
    hdu.writeto(cube.infile, overwrite=True)

    if filename_q3di is not None:
        # argslinefit
        q3di_object.argslinefit = argslinefit_original

        np.save(filename_q3di, q3di_object)


# ======================================
def get_spiral_spaxels(q3di, col_spiral_center=None, row_spiral_center=None,
                       cols=None, rows=None, cube=None):

    q3di1 = q3dutil.get_q3dio(q3di)

    # cube
    if cube is None:
        cube = q3di1.load_cube()

    # 2D grid
    cols2D, rows2D = np.array(np.meshgrid(np.arange(q3di1.ncols),
                                          np.arange(q3di1.nrows)))# + 1
    # To matched the tranverse orientation of q3dfit
    cols2D, rows2D = cols2D.T, rows2D.T

    # Use only good spaxels
    if cols is None:
        cols = [1, q3di1.ncols]
    if rows is None:
        rows = [1, q3di1.nrows]

    # center
    if col_spiral_center is None:
        col_spiral_center = int(q3di1.ncols/2)
    if row_spiral_center is None:
        row_spiral_center = int(q3di1.nrows/2)

    # slices
    row_slice = slice(rows[0]-1, rows[1])
    col_slice = slice(cols[0]-1, cols[1])
    slice_cube = (col_slice, row_slice)
    #slice_cube = (col_slice, row_slice)

    # 1D
    cols1D = cols2D[slice_cube].ravel()
    rows1D = rows2D[slice_cube].ravel()

    # only good spaxels
    spaxgood = ~(cube.dq.astype(bool).all(axis=-1))
    spaxgood = spaxgood[slice_cube].ravel()
    #spaxgood = spaxgood[cols[0]-1:cols[1], rows[0]-1:rows[1]].ravel()
    cols1D, rows1D = cols1D[spaxgood], rows1D[spaxgood]

    if len(cols1D) > 0:
        # Radius
        radius = np.sqrt((cols1D - (col_spiral_center-1))**2 +\
                         (rows1D - (row_spiral_center-1))**2)

        # Theta
        theta = np.arctan2(rows1D - (row_spiral_center-1),
                           cols1D - (col_spiral_center-1))

        # Spiral map
        spiral_map = radius + 0.001*((theta - theta.min()) / (theta.max()-theta.min()))

        # Sort the spaxels by radius flattened
        idx_spiral = np.argsort(spiral_map.ravel())

        # output
        nspax = len(idx_spiral)
        colarr = cols1D[idx_spiral]
        rowarr = rows1D[idx_spiral]
        #plt.plot(cols1D[idx_spiral][:2000], rows1D[idx_spiral][:2000])

        return nspax, colarr, rowarr

    else:
        return 0, None, None


# ======================================
# -------- Functions -------------------
def gaussian(x, *p):
    y = x*0
    for i in range(len(p)//3):
        y = y + p[0 + 3*i] * np.exp(-0.5*((x-p[1+3*i])/p[2+3*i])**2)
    return y

# ===============================
# ----------- utils -------------
def eval_quantity(str_quantity):
    '''
    Convert string to astropy.units.Quantity
    
    Input
    -----
    str_quantity: str
        String of the quantity. Units should be separated by a space from the
        value.
        Example of accepted inputs:
            15 arcsec
            [15, 14,5] arcsec
            15 * arcsec
            [15, 14,5] * arcsec
        Not accepted:
            15arcsec

    Returns
    -------
    quantity: astropy.units.Quantity
        Evaluated quantity with units.
    '''

    if '*' in str_quantity:
        str_value, str_unit = str_quantity.strip(' ').split("*")
        #print(str_value, str_unit)
        value = json.loads(str_value)
        unit = u.Unit(str_unit)

    else:
        split_str_quantity = str_quantity.strip(' ').split(' ')
        try:
            unit = u.Unit(split_str_quantity[-1])
            value = json.loads(' '.join(split_str_quantity[:-1]))
        except:
            unit = 1
            value = json.loads(' '.join(split_str_quantity))
        #print(unit)

    if value is None:
        return value
    else:
        return value * unit

def is_iterable(obj, check_not_str=True, check_not_header=True):
    '''
    check_str: Also check if it is not str?
               Remender that str is iterable by itself.
    '''
    #hasattr(obj, '__iter__'): FIXME: test this option

    try:
        null = iter(obj)
        logic = True
    except TypeError:
        logic = False

    # Check if it is not str (which is also iterable)
    if check_not_str:
        logic = logic and not isinstance(obj, str)
    
    if check_not_header:
        logic = logic and not isinstance(obj, astropy.io.fits.Header)

    return logic

