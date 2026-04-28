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

def get_opt_load_maps(cfg):
    from astropy import units as u

    opts_load_maps = dict(SNR_cut=cfg.getfloat('load_fits', 'SNR_cut'),
                            which_snr_selection=cfg.get('load_fits', 'which_snr_selection'),
                            which_pkflux_noise=cfg.get('load_fits', 'which_pkflux_noise'),
                            fluxpk_noise_fitsfile=cfg.get('load_fits', 'fluxpk_noise_fitsfile'),
                            which_flux_noise=cfg.get('load_fits', 'which_flux_noise'),
                            apply_mask=cfg.getboolean('load_fits', 'apply_mask'),
                            stats=cfg.getboolean('load_fits', 'stats'),
                            redchisqmax=cfg.getfloat('load_fits', 'redchisqmax'),
                            sigmin=bq3dutils.read_init(cfg, 'load_fits', 'sigmin',unit='km/s', return_iterable=False),
                            velmin=bq3dutils.read_init(cfg, 'load_fits', 'velmin',unit='km/s', return_iterable=False),
                            velmax=bq3dutils.read_init(cfg, 'load_fits', 'velmax',unit='km/s', return_iterable=False),
                            type_mask2D=cfg.get('load_fits', 'type_mask2D'),
                            transpose=cfg.get('load_fits', 'transpose')
                            )
    return opts_load_maps

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
        q3di.filename_noise = q3di.filename_noise.replace(rootdir_old, rootdir_new)
        q3di.filename_q3di_continuum = q3di.filename_q3di_continuum.replace(rootdir_old, rootdir_new)

        np.save(filename_q3di, q3di)

    compact_directory_targz(dirname, filename_targz)

    # ================
    # Reverse
    #q3di = q3dutil.get_q3dio(filename_q3di)
    q3di.outdir = q3di.outdir.replace(rootdir_new, rootdir_old)
    q3di.infile = q3di.infile.replace(rootdir_new, rootdir_old)
    q3di.logfile = q3di.logfile.replace(rootdir_new, rootdir_old)
    q3di.filename_noise = q3di.filename_noise.replace(rootdir_new, rootdir_old)
    q3di.filename_q3di_continuum = q3di.filename_q3di_continuum.replace(rootdir_new, rootdir_old)

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

def read_init(cfg_fit, section, key, unit='km/s', return_iterable=True):
    value_init = json.loads(cfg_fit.get(section, key))
    if value_init is not None:
        if return_iterable:
            if is_iterable(value_init):
                value_init = np.array(value_init) * u.Unit(unit)
            else:
                value_init = np.array([value_init]) * u.Unit(unit)
            return value_init
        else:
            value_init = value_init * u.Unit(unit)
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
    print('WARNING: Assuming uniformily spaced spectral axis for continuum img')
    diffwave_set = np.array(list(set(np.diff(wave.value))))
    #assert len(diffwave_set) == 1, f"Not uniform spectral axis: {diffwave_set}"
    diffwave_set_pct = (abs(diffwave_set / diffwave_set[0] -1 ) * 100) < 0.1
    assert diffwave_set.all(), f"Not uniform spectral axis: {diffwave_set_pct}"
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

    if filename_noise is None:
        filename_noise = q3di_continuum.filename_noise
    #print(f'filename_noise = {filename_noise}')

    #skip_cont = False
    if 1:
        hdu_noise = fits.open(filename_noise)
        if len(hdu_noise) == 1:
            
            #raise Exception("Check")
            skip_cont = True
            #noise = fits.getdata(filename_noise).T
            noise = read_quantity(filename_noise, 0)

            #if skip_cont:
            #    return noise * np.nan
            #if 0:
            #    noise = fits.getdata(filename_noise)  * u.Unit('erg / (cm2 s um)')
            uncertainty_flux_cont = np.sqrt(float(len(wave))) * (noise * dw).to('erg / (cm2 s)')

        else:
            # Read the polynomial fit noise data from the specified file
            noise_polyfit = read_quantity(filename_noise, 'polyfit')

            # Get the spectral axis from the header and convert it to match the wave unit
            he_polyfit = hdu_noise['polyfit'].header
            wave_polyfit = get_spectral(he_polyfit).to(wave.unit)

            # Filter the polynomial fit wavelengths and noise data to the valid range
            wave0, wave1 = wave[0] - dw * 0.1, wave[-1] + dw * 0.1

            good_wave_polyfit = (wave_polyfit >= wave0) & (wave_polyfit <= wave1)

            wave_polyfit = wave_polyfit[good_wave_polyfit]
            noise_polyfit = noise_polyfit[good_wave_polyfit]

            # Calculate the uncertainty in the total flux by integrating the noise over the wavelength range
            uncertainty_flux_cont = np.sqrt(np.sum((noise_polyfit * dw) ** 2)).to('erg / (cm2 s)')

        mask_snr = (flux_cont.data / uncertainty_flux_cont) < SNR_cut

        flux_cont.mask += mask_snr


    # flux densityy
    if flux_density:
        
        flux_cont_density = np.ma.array(flux_cont.data / bandwidth,
                                        mask=flux_cont.mask)

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


#from bcodes import bfit
def get_noise_lines(q3di=None, cubefile_for_rms=None, idx_speclist=None,
                    filename_noise=None, lines=None, redshift=None,
                    filename_dat=None, used_nosie1D=False, poly_deg_spectral=4,
                    poly_deg_vertical=None, poly_deg_horizontal=None):

    if q3di is not None:
        if lines is None:
            lines = q3di.lines
        if cubefile_for_rms is None:
            cubefile_for_rms = q3di.infile
        if redshift is None:
            redshift = q3di.zsys_gas

        if filename_noise is None:
            filename_noise = '{}/aux/{}'.format(os.path.dirname(cubefile_for_rms),
                                                os.path.basename(cubefile_for_rms).replace('.fits', '_noiselines.fits'))

    assert cubefile_for_rms is not None, "Inform cubefile_for_rms"

    data = read_quantity(cubefile_for_rms, ext='SCI')
    wave = get_spectral(fits.getheader(cubefile_for_rms, 'SCI'))

    linelist = q3dutil.linelist(lines, vacuum=True)

    linelist['lines_obs'] = linelist['lines'] * (1 + redshift)
    linelist['lines_pix'] = (((linelist['lines_obs'] - wave[0]) / np.diff(wave)[0]).astype(int)).value

    bunit = data.unit
    data = np.ma.array(data.value, mask=np.isnan(data.data))

    data_std2D = []
    #datai_mean = []
    #datai_std1D = []
    #datai_mean1D = []
    gauss_mean, gauss_std = [], []
    region_wave = []
    wave_center_list = []
    for i, idx in enumerate(idx_speclist):
        wave_center = wave[int(np.mean(idx))]
        wave_center_list += [wave_center]
        print(i, len(idx_speclist), f'{wave_center:.2f}')
        region_wave += [wave[int(np.mean(idx))]]

        datai = data[idx[0]:idx[1]]
        data_std2D += [np.nanstd(datai, axis=0)]
        #datai_std1D += [np.nanstd(datai)]
        #datai_mean += [np.nanmean(datai, axis=0)]

        indir_hist_pngfile = f'{os.path.dirname(cubefile_for_rms)}/aux/png/'
        if not os.path.isdir(indir_hist_pngfile):
            os.mkdir(indir_hist_pngfile)

        hist_pngfile = f'{indir_hist_pngfile}/{os.path.basename(cubefile_for_rms)}_{wave_center.value:.2f}{wave_center.unit}.png'

        p_gauss = get_rms_mean(datai, plot=1, threshold=2, n=100,
                               outfile=hist_pngfile,
                               show=False)
        
        gauss_mean += [p_gauss[1]] 
        gauss_std += [p_gauss[2]]


    #datai_std1D = np.array(datai_std1D)
    region_wave = u.Quantity(region_wave)
    gauss_mean = np.array(gauss_mean)
    gauss_std = np.array(gauss_std)
    data_std2D = np.array(data_std2D)
    wave_center_list = u.Quantity(wave_center_list)
    #datai_mean = np.array(datai_mean)

    ppoly, pstdpoply = fit_polynomial(region_wave.value, gauss_std,
                                       deg=poly_deg_spectral,
                                       plot=0, niter=4)
 
    # plot radial rms
    polyfit = np.polyval(ppoly, region_wave.value)

    if 1:
        #print(region_wave, polyfit)
        plt.plot(region_wave, gauss_std, 'o-', color='C3', label='gaussian std (50 pix width)')
        plt.plot(region_wave, polyfit, color='k', label='polyfit (deg=4)')
        plt.xlabel('wave (um)')
        plt.ylabel(f'Continuum standard deviation ({bunit.to_string()})')
        plt.legend(loc='best', fontsize=8)

        for line_obs, line_name in zip(linelist['lines_obs'], linelist['linelab']):
            plt.axvline(line_obs, color='grey', linestyle=':', alpha=0.3)
            y_position = plt.ylim()[1] * .95
            # Adjust y_position to avoid overlap
            offset = 0.02 * plt.ylim()[1]  # Adjust offset as needed
            y_position -= offset * linelist['lines_obs'].tolist().index(line_obs)
            plt.text(line_obs, y_position, line_name, color='k', fontsize=6, rotation=0, horizontalalignment='left')
 
        pngfile = filename_noise.replace('.fits', '.png')

        plt.savefig(pngfile, bbox_inches='tight')


    if 1:

        data_std2D = np.ma.array(data_std2D, mask=data_std2D==0)
        nidx, ny, nx = data_std2D.shape

        for i in range(nidx):
            data_std2D_i = data_std2D[i]

            plt.close('all')

            fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
            ax1, ax2 = ax
            if 1:
                # mean along - x  (vertical variation)
                data_std2D_i_vertical_mean_along_x = np.mean(data_std2D_i, axis=1)

                for ix in range(nx):
                    ax1.plot(data_std2D_i[:, ix], color='k', lw=1, alpha=.5)

                ax1.plot(data_std2D_i_vertical_mean_along_x, color='C3', lw=2, alpha=.7)

            if 1:
                # mean along - y (horizontal variation)
                data_std2D_i_horizotal_mean_along_y = np.mean(data_std2D_i, axis=0)
                for ix in range(ny):
                    ax2.plot(data_std2D_i[ix, :], color='k', lw=1, alpha=.5)

                ax2.plot(data_std2D_i_horizotal_mean_along_y, color='C3', lw=2, alpha=.7)

            opts_title = {'fontsize': 12, 'verticalalignment': 'top', 'y': 0.90}
            ax1.set_title('vertical variation', **opts_title)
            ax2.set_title('horizontal variation', **opts_title)

            ax2.set_xlabel('x/y pixel')        

            axFig = bq3dplots.get_axFig(fig)

            axFig.set_title(f'{wave_center}')
            axFig.set_ylabel(f'Continuum std ({bunit.to_string()})', labelpad=15)

            fig.tight_layout()

            xy_pngfile = f'{indir_hist_pngfile}/mean_xy_{os.path.basename(cubefile_for_rms)}_{wave_center.value:.2f}{wave_center.unit}.png'

            fig.savefig(xy_pngfile, bbox_inches='tight')

        if 1:
            data_std2D_horizotal_mean_along_y = np.mean(data_std2D, axis=1)
            data_std2D_vertical_mean_along_x = np.mean(data_std2D, axis=2)

            std_polyfit = np.polyval(ppoly, wave_center_list.value)
            data_std2D_mean_mean_normed_vertical = np.mean(data_std2D_vertical_mean_along_x / std_polyfit[:,np.newaxis], axis=0)
            data_std2D_mean_mean_normed_horizontal = np.mean(data_std2D_horizotal_mean_along_y / std_polyfit[:,np.newaxis], axis=0)

            if poly_deg_vertical is not None:
                ppoly_vertical, pstdpoply_vertical = fit_polynomial(
                    np.arange(ny), data_std2D_mean_mean_normed_vertical, 
                    deg=poly_deg_vertical, plot=1, niter=1, lower_threshold=4,
                     upper_threshold=4)
                
                polyfit_vertical = np.polyval(ppoly_vertical, np.arange(ny))

            if poly_deg_horizontal is not None:
                raise Exception("Not implemented")

            plt.close('all')

            from matplotlib import cm
            norm = plt.Normalize(vmin=wave_center_list.min().value, vmax=wave_center_list.max().value)
            cmap = cm.get_cmap('viridis')

            fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
            ax1, ax2 = ax

            for i, wave_center in enumerate(wave_center_list):
                wave_center = wave_center_list[i]
                std_polyfit_i = np.polyval(ppoly, wave_center.value)

                color = cmap(norm(wave_center.value))

                ax1.plot(data_std2D_vertical_mean_along_x[i] / std_polyfit_i, color=color, lw=1., alpha=.7, label=f'{wave_center:.2f}')
                ax2.plot(data_std2D_horizotal_mean_along_y[i] / std_polyfit_i, color=color, lw=1., alpha=.7, label=f'{wave_center:.2f}')

            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Wave Center (um)', location='right')

            ax1.plot(data_std2D_mean_mean_normed_vertical, color='C3', lw=2, alpha=1)
            ax2.plot(data_std2D_mean_mean_normed_horizontal, color='C3', lw=2, alpha=1)

            ax1.plot(polyfit_vertical, color='k', lw=2, alpha=.7)

            opts_title = {'fontsize': 12, 'verticalalignment': 'top', 'y': 0.90}
            ax1.set_title('vertical variation', **opts_title)
            ax2.set_title('horizontal variation', **opts_title)

            ax2.set_xlabel('x/y pixel')        

            axFig = bq3dplots.get_axFig(fig)

            axFig.set_ylabel(f'Continuum std ({bunit.to_string()})', labelpad=15)


            fig.tight_layout()

            xy_pngfile = f'{indir_hist_pngfile}/mean_xy_fit_{os.path.basename(cubefile_for_rms)}_{wave_center.value:.2f}{wave_center.unit}.png'

            fig.savefig(xy_pngfile, bbox_inches='tight')


        #for i in range(nidx):
        #    plt.plot(data_std2D_horizotal_mean_along_y[i])


    # Plot all maps from data_std2D in a grid of subplots
    if 1:
        plt.close('all')
        nrows = int(np.sqrt(nidx))
        ncols = int(np.ceil((nidx / nrows)))
            
        aspect_ratio = ncols / nrows
        fig_width = 6
        fig_height = fig_width / aspect_ratio
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=True)

        axes = axes.ravel()

        for ax, i in zip(axes, np.arange(nidx)):
            if i < len(data_std2D):
                im = ax.imshow(data_std2D[i], origin='lower', cmap='viridis')
                ax.set_title(f'{wave_center_list[i]:.2f}', y=0.7)
            else:
                ax.axis('off')  # Turn off unused subplots

        # Remove ticks/labels for axes that are not on the edges
        for i, ax in enumerate(axes):
            if i % ncols != 0:  # Not in the first column
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i < (nrows - 1) * ncols:  # Not in the last row
                ax.set_xticklabels([])
                ax.set_xlabel("")
        
        plt.tight_layout()
        pngfile = filename_noise.replace('.fits', '_2Dmaps_input.png')
        plt.savefig(pngfile, bbox_inches='tight')
        plt.close('all')
        #plt.show()

    # Save it to dat file
    if filename_dat is None:

        filename_dat = filename_noise.replace('.fits', '.dat')

    with open(filename_dat, 'w') as arq:
        arq.write(f'# wave({wave.unit}), gauss_std ({bunit}),polyfit ({bunit})\n')
        for i in range(len(region_wave)):
            arq.write(f'{region_wave[i]:.4f},{gauss_std[i]:.4e},{polyfit[i]:.4e}\n')

    #noise_2D = np.ones((ny, nx)) * polyfit_vertical[:, np.newaxis]


    noise_poly2d = np.ones((ny, nx))
    if poly_deg_vertical is not None:
        noise_poly2d *= polyfit_vertical[:, np.newaxis]
        #print(1111111111111, poly_deg_vertical)

    if poly_deg_horizontal is not None:
        raise Exception("Not implemented")
        noise_poly2d *= poly_deg_horizontal[:, np.newaxis]


    # Write it to fitsfile
    noise_line = {}
    #dq = fits.getdata(cubefile_contsub)
    mask2D = np.isnan(data.data).astype(bool).all(axis=0)
    for line in lines:
        iline_wave_obs = linelist['lines_obs'][linelist['name'] == line][0]
        iline_noise_poly = np.polyval(ppoly, iline_wave_obs)
        
        if used_nosie1D:
            iline_noise_poly2d = np.zeros_like(data.data[0]) + iline_noise_poly
            iline_noise_poly2d[mask2D] = np.nan
        else:
            # vertical
            iline_noise_poly2d = noise_poly2d * iline_noise_poly
            iline_noise_poly2d[mask2D] = np.nan

        noise_line[line] = iline_noise_poly2d

    data_exts = []
    name_exts = []
    for line in lines:
        data_exts += [noise_line[line]]
        name_exts += [f'{line}']

    header3D = fits.getheader(cubefile_for_rms, 'SCI')

    #print(2222, filename_noise)
    write_multifits(data_exts, filename_noise, header3D, name_exts,
                            overwrite=True)

    # add polyfit / 2D variations
    with fits.open(filename_noise) as hdu:
        # Add the polynomial coefficients to the header

        # Save the modified file

        header3 = get_wcs(header3D).spectral.to_header()
        header3['bunit'] = bunit.to_string()


        polyfit_full = np.polyval(ppoly, wave.value)

        #print(len(polyfit_full), len(wave))


        hdu_polyfit = fits.ImageHDU(data=polyfit_full, header=header3,
                                    name='polyfit', )

        header_poly2D = get_wcs(header3D).celestial.to_header()
        header_poly2D['bunit'] = ''
        hdu_polyfit2D = fits.ImageHDU(data=noise_poly2d, header=header_poly2D,
                                     name='polyfit2D')


        hdu.append(hdu_polyfit)
        hdu.append(hdu_polyfit2D)

        assert poly_deg_horizontal is None
        hdu[0].header['HISTORY'] =  "Noise variation along spectral dimention obtained from the standard deviation" +\
                                    f" in line-free regions (50-pixel wide), and later modeled with a"+\
                                    f" {poly_deg_spectral}-degree polynomial. fit. " +\
                                    f" Noise spatial variation dependence along the y-axis modeled with a" +\
                                    f" {poly_deg_vertical}-degree polynomial" +\
                                    f" ({np.datetime64('today', 'D')} by Dall'Agnol de Oliveira, B.)"

        hdu.writeto(filename_noise, overwrite=True)
        print(hdu.info())

    return gauss_mean, gauss_std



# LINEDATA
class bLineData:
    '''
    Read in and store line data for all lines found in *.lin.npz file
    (which is output by q3da).

    Individual line measurements can be obtained with corresponding methods.

    Parameters
    -----------
    q3di : object

    Attributes
    ----------
    lines : dict
        Line names.
    maxncomp : int
        Maximum number of components fit to a line.
    data : dict
        Contents of the line data (.npz) file.
    Examples
    --------
    >>>

    Notes
    -----
    '''

    def __init__(self, q3di, datafile=None):

        if datafile is None:
            filename = q3di.label+'.line.npz'
            datafile = os.path.join(q3di.outdir, filename)
            #input_datafile = False
        #else:
        #    input_datafile = True

        print(datafile)
        # print(datafile)
        if not os.path.exists(datafile):
            print('ERROR: emission line ('+filename+') file does not exist')
            return
        self.lines = q3di.lines
        self.data = self.read_npz(datafile)
        
        #self.maxncomp = q3di.maxncomp
        self.maxncomp = len([i for i in self.data['emlweq'].item().keys() if 'fc' in i])

        # book-keeping inheritance from initproc
        self.ncols = self.data['ncols'].item()
        self.nrows = self.data['nrows'].item()
        self.bad = np.nan
        self.dataDIR = q3di.outdir
        self.target_name = q3di.name
        # self.flux    = self.get_flux()
        # self.siga    = self.get_sigma()
        # self.wavelen = self.get_wave()
        # self.eq      = self.get_weq()
        return

    def read_npz(self, datafile):
        ''' Load binary line data file.

        Parameters
        ----------
        datafile : str

        Returns
        -------
        Contents of datafile.

        '''
        dataread = np.load(datafile, allow_pickle=True)
        self.colname = sorted(dataread)
        return dataread

    def get_flux(self, lineselect, FLUXSEL='ftot'):
        ''' Get flux and error of a given line.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        fluxsel : str, default 'ftot' (total flux)
            Which flux to grab. String names defined in q3da.

        Returns
        -------
        dict
            keys flux, fluxerr contain ndarray(ncols, nrows, ncomp)

        '''

        # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1', 'fc1pk')
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        emlflx = self.data['emlflx'].item()
        emlflxerr = self.data['emlflxerr'].item()
        dataout = {'flux': emlflx[FLUXSEL][lineselect],
                   'fluxerr': emlflxerr[FLUXSEL][lineselect]}
        return dataout

    def get_ncomp(self, lineselect):
        ''' Get # components fit to a given line.

        Parameters
        ----------
        lineselect : str

        Returns
        -------
        ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        return (self.data['emlncomp'].item())[lineselect]

    def get_sigma(self, lineselect, COMPSEL=1):
        ''' Get sigma and error of a given line and component.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        compsel : int, default 1

        Returns
        -------
        dict
            keys sig, sigerr contain ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlsig = self.data['emlsig'].item()
        emlsigerr = self.data['emlsigerr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'sig': emlsig[csel][lineselect],
                   'sigerr': emlsigerr[csel][lineselect]}
        return dataout

    def get_wave(self, lineselect, COMPSEL=1):
        ''' Get central wavelength and error of a given line and component.

        Parameters
        ----------
        lineselect : str
            Which line to grab.
        compsel : int, default 1

        Returns
        -------
        dict
            keys wav, waverr contain ndarray(ncols, nrows)

        '''
        if lineselect not in self.lines:
            print('ERROR: line does not exist')
            return None
        # 'c1'
        emlwav = self.data['emlwav'].item()
        emlwaverr = self.data['emlwaverr'].item()
        csel = 'c'+str(COMPSEL)
        dataout = {'wav': emlwav[csel][lineselect],
                   'waverr': emlwaverr[csel][lineselect]}
        return dataout

    # def get_weq(self, lineselect, FLUXSEL='ftot'):
    #     # FLUXSEL = 'ftot' by default --> select from ('ftot', 'fc1')
    #     if lineselect not in self.lines:
    #         print('ERROR: line does not exist')
    #         return None
    #     # 'ftot', 'fc1'
    #     emlweq = self.data['emlweq'].item()
    #     dataout = emlweq[FLUXSEL][lineselect]
    #     return dataout


# ----------- fit -------
# ---------------------------- Fit --------------------------
def fit_gaussian_hist(x, freq, dx=None, p0=None, plot=False):
    '''
    x:
    freq: 
    '''

    from scipy.optimize import curve_fit

    mask = np.isnan(freq)
    freq = freq[~mask]
    x = x[~mask]
    norm = freq.max()
    norm_x = x.mean()
    #freq = freq / norm
    #x = x / norm_x

    def gauss(x, *p):
        y = x*0
        for i in range(len(p)//3):
            y = y + p[0 + 3*i] * np.exp(-0.5*((x-p[1+3*i])/p[2+3*i])**2)
        return y

    if p0 is None:
        x_sum = np.trapz(freq, x)
        if dx is None:
            dx = np.diff(x)
            dx = np.append(dx, dx[-1])

        x_mean = np.sum((x) * freq * dx) / x_sum
        x_std = np.sqrt(np.sum((x-x_mean)**2 * freq * dx) / x_sum)

        mask_x = (x > (x_mean - 0.5*x_std)) & (x < (x_mean + 0.5*x_std))

        if np.sum(mask_x):
            freq_max = np.max(freq[mask_x])
        else:
            freq_max = np.max(x)

        p0 = [freq_max, x_mean, x_std]

    p, cov = curve_fit(gauss, x, freq, p0=p0)

    return p, cov

def fit_polynomial(x0, y0, w0=None, deg=1, plot=False, niter=1,
                   lower_threshold=2, upper_threshold=2, figname=None, 
                   percentage_ncut_threshold=3, xy_final=False):

    #x = deepcopy(x0)
    #y = deepcopy(y0)
    x = np.ma.array(x0)
    y = np.ma.array(y0)
    nanmask = np.isnan(x) | np.isnan(y)
    mamask = x.mask | y.mask
    mask0 = mamask + nanmask
    x = x[~mask0].data
    y = y[~mask0].data

    w = deepcopy(w0)
    if w is not None:
        w = np.ma.array(w)
        w = w[~mask0]
                
    
    n = len(y)
    for i in range(niter):
        p, cov = np.polyfit(x, y, w=w, deg=deg, cov=1)
        f = np.polyval(p, x)

        r = y - f
        std = np.std(r)

        good = (r < upper_threshold*std) & (r > -lower_threshold*std)

        percentage_ncut = 100*(1-len(y[good])/(len(y)))
        percentage_fcut = sum((r/std)**2/(len(y)-deg))#sum((f - f[mask])**2)
        #print(percentage_ncut, percentage_fcut)
        if percentage_ncut < percentage_ncut_threshold:
            break
        y = y[good]
        x = x[good]
        f = f[good]
        if w is not None:
            w = w[good]

    if plot:
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.plot(x0, y0, '.')
        plt.plot(x, y, '.')
        xfit = np.sort(x)
        plt.plot(xfit, np.polyval(p, xfit), '-')

    sigma_p = np.sqrt(np.diag(cov))
    if xy_final:
        return p, sigma_p, x, y
    else:
        return p, sigma_p

def get_rms_mean(datax, x1x2y1y2=None, n=250, plot=False, niter=2,
                 threshold=3, n_rms=5, outfile=None, show=False):

    if x1x2y1y2 is not None:
        x1, x2, y1, y2 = x1x2y1y2

        x = deepcopy(datax[x1:x2, y1:y2].ravel())
    else:
        x = deepcopy(datax)

    x = np.ma.array(x)
    mask = x.mask + np.isnan(x)
    x = x[~mask].ravel()

    mean, rms = np.nanmean(x), np.nanstd(x)
    for i in range(niter):

        x = x[(abs(x - mean) / rms) < threshold]

        k = n / (n_rms * 2)
        bins = np.arange( mean - rms*n_rms, mean + rms*n_rms, rms/k)

        freq, bins = np.histogram(x, bins)
        xbins = bins[1:] - 0.5*(bins[1] - bins[0])

        #from bcodes import bfit
        p, e_p = fit_gaussian_hist(xbins[freq != 0], freq[freq != 0])
        mean = p[1]
        rms = p[2]
        A = p[0]
      
    if plot:
        plt.figure()
        plt.hist(x, bins, density=False)
        plt.plot(xbins, gaussian(xbins, *p))
        
        #bins1 = np.arange(mean - rms*n_rms*2, mean + rms*n_rms*2, rms/k/2)
        plt.hist(datax[~mask].ravel(), bins, alpha=.3, zorder=-2, density=False)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(plt.gcf())

        
    return A, mean, rms

# --------- Q3D - maps ------------------------
#from q3dfit.q3dpro import OneLineData, ContData, LineData
def load_linemaps_all(q3di, **opts_load_maps):

    opts_load_maps1 = {'cube':None}
    opts_load_maps1.update(opts_load_maps)

    if 'line' in opts_load_maps1:
        opts_load_maps1.pop('line')

    # Load q3di data
    q3di = q3dutil.get_q3dio(q3di)

    # Get linelist and linenames
    linelist = q3dutil.get_linelist(q3di)
    linenames = linelist['name'].value

    # Load cube data
    if opts_load_maps1['cube'] is None:
        cube = q3di.load_cube()
        opts_load_maps1['cube'] = cube

    # load individual maps into dict
    linemaps = {}
    for line in linenames:
        linemaps[line] = load_linemaps(q3di, line, **opts_load_maps1) 

    # create master map for stats involving all lines
    mask_line_ar = np.ma.array([linemaps[line].redchisq.mask for line in linenames])
    mask_alllines = ~(~mask_line_ar).any(axis=0)

    # update stats with the new mask
    keylist = ['redchisq', 'dof', 'ndata', 'maxncomp',
               'chisq', 'nparsfree', 'logLmax0', 'aic0', 'bic0',  'logLmax',
               'aic', 'bic']


    line_ref = linenames[0]
    if 1:
        # Update individually
        for key in keylist:

            data_key = getattr(linemaps[line_ref], key)
            data_key.mask = mask_alllines
            for line in linenames:
                linemaps[line]
                setattr(linemaps[line], key, data_key)
    else:
        # Add new key to dicts 
        for key in keylist:

            data_key = deepcopy(getattr(linemaps[line_ref], key))
            data_key.mask = mask_alllines

            linemaps[key] = data_key

    return linemaps


def load_linemaps(q3di, line='[OIII]5007', cube=None, transpose=True,
                  stats=True, type_mask2D='any', sigmin=None,
                  velmin=None, velmax=None,
                  redchisqmax=None,
                  apply_mask=True, 
                  SNR_cut=3, SNR_cut_low=2.5,
                  SNR_cut_flux=None,
                  redshift=None, 
                  which_snr_selection='pkflux',
                  which_pkflux_noise='fitsfile',
                  fluxpk_noise_fitsfile=None,
                  which_flux_noise='lmfit',
                  filename_npz=None,
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
    line : str, optional
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

    if filename_npz is not None:
        linedat = bLineData(q3di, datafile=filename_npz)
    else:
        linedat = LineData(q3di)
    linedata = OneLineData(linedat, line)


    #print(f'ncomp = {linedat.maxncomp}')
    # get errors
    def get_errors(self, linedat):

        self.fluxerr = \
            np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
                     dtype=float) + linedat.bad
        self.pkfluxerr = \
            np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
                     dtype=float) + linedat.bad
        self.sigerr = \
            np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
                     dtype=float) + linedat.bad
        self.waveerr = \
            np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
                     dtype=float) + linedat.bad
        
        #if 0:
        #    self.pkflux_obs = \
        #        np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
        #                dtype=float) + linedat.bad
        #    self.sig_obs = \
        #        np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
        #                dtype=float) + linedat.bad#

        #    self.pkfluxerr_obs = \
        #        np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
        #                dtype=float) + linedat.bad
        #    self.sigerr_obs = \
        #        np.zeros((linedat.ncols, linedat.nrows, linedat.maxncomp),
        #                dtype=float) + linedat.bad

        if 1:
            from astropy.stats import gaussian_sigma_to_fwhm

            if type_mask2D == 'any':
                fitted2D =  ~np.isnan(self.flux).T.any(axis=0)
            elif type_mask2D == 'all':
                fitted2D =  ~np.isnan(self.flux).T.all(axis=0)

            fitted3D = ~np.isnan(self.flux)
            #fitted2D = fitted3D
            
            y_ref, x_ref,  = np.argwhere(fitted2D)[0]
            y_row_ref, x_col_ref = y_ref + 1, x_ref + 1
            q3do_ref = load_q3dout(q3di, x_col_ref, y_row_ref)

            specres_ref = [item_ref for key_ref, item_ref in q3do_ref.param.items() if 'SPECRES' in key_ref][0]

            Ruse_3D = np.zeros_like(self.flux)
            Ruse_3D[fitted3D] = np.vectorize(specres_ref.get_R)(self.wave[fitted3D])
            
            #fitted3D = ~np.isnan(self.flux)).any(axis=0)
            #np.vectorize(specres_ref.get_R)(self.wave[fitted2D])

            self.sig_obs = \
                np.sqrt(self.sig**2 +
                        (c_kms.to('km/s').value / Ruse_3D / 
                            gaussian_sigma_to_fwhm)**2)
            
            self.sigerr_obs = self.sigerr * \
                self.sig_obs / self.sig
            

            self.pkflux_obs = self.pkflux * \
                self.sig / self.sig_obs

            self.pkfluxerr_obs = self.pkfluxerr # (* self.sig / self.sig_obs)??


        for i in range(0, linedat.maxncomp):
            self.fluxerr[:, :, i] = \
                (linedat.get_flux(line, FLUXSEL='fc'+str(i+1)))['fluxerr']
            self.pkfluxerr[:, :, i] = \
                (linedat.get_flux(line,
                                   FLUXSEL='fc'+str(i+1)+'pk'))['fluxerr']
            self.sigerr[:, :, i] = \
                (linedat.get_sigma(line, COMPSEL=i+1))['sigerr']
            self.waveerr[:, :, i] = \
                (linedat.get_wave(line, COMPSEL=i+1))['waverr']
            
            #if 0:
            #    self.pkflux_obs[:, :, i] = \
            #        (linedat.get_flux(line,
            #                        FLUXSEL='fc'+str(i+1)+'pk'))['flux_obs']
            #    self.sigerr_obs[:, :, i] = \
            #        (linedat.get_sigma(line, COMPSEL=i+1))['sig_obs']
            #    
            #    self.pkflux_obs[:, :, i] = \
            #        (linedat.get_flux(line,
            #                        FLUXSEL='fc'+str(i+1)+'pk'))['fluxerr_obs']
            #    self.sigerr_obs[:, :, i] = \
            #        (linedat.get_sigma(line, COMPSEL=i+1))['sigerr_obs']
        return self


    linedata = get_errors(self=linedata, linedat=linedat)

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

    linedata.pkflux_obs = linedata.pkflux_obs * cube.fluxnorm * bunit
    linedata.pkfluxerr_obs = linedata.pkfluxerr_obs * cube.fluxnorm * bunit

    # Velocity Dispersion
    linedata.sig = linedata.sig * u.kms
    linedata.sigerr = linedata.sigerr * u.kms

    linedata.sig_obs = linedata.sig_obs * u.kms
    linedata.sigerr_obs = linedata.sigerr_obs * u.kms
    
    # Central wavelength of the fitted Gaussian
    linedata.wave = linedata.wave * sunit
    linedata.waveerr = linedata.waveerr * sunit

    # Flux
    linedata.flux = linedata.flux * cube.fluxnorm * funit
    linedata.fluxerr = linedata.fluxerr * cube.fluxnorm * funit

    # Radial Velocity (in km/s)
    linelist = q3dutil.get_linelist(q3di)
    rest_line = linelist[linelist['name'] == line]['lines'][0] * u.um
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
        if fluxpk_noise_fitsfile in ['', None]:
            if hasattr(q3di, 'filename_noise'):
                fluxpk_noise_fitsfile = q3di.filename_noise
            
        if fluxpk_noise_fitsfile is None:
            raise Exception(f'fluxpk_noise_fitsfile not found: {fluxpk_noise_fitsfile}')
        


        # pkfluxerr_for_snr2D = fits.getdata(fluxpk_noise_fitsfile).T * bunit# * cube.fluxnorm * bunit
        hdu_noise = fits.open(fluxpk_noise_fitsfile)
        if len(hdu_noise) == 1:
            #pkfluxerr_for_snr2D = fits.getdata(fluxpk_noise_fitsfile).T
            pkfluxerr_for_snr2D = read_quantity(fluxpk_noise_fitsfile, 0).T
        else:
            assert line in hdu_noise
            pkfluxerr_for_snr2D = read_quantity(fluxpk_noise_fitsfile, line).T

        # Tile the 2D error to match the 3D shape of the flux data
        linedata.pkfluxerr_for_snr = np.tile(pkfluxerr_for_snr2D.T, 
                                             (linedata.flux.shape[2], 1, 1)).T
    else:
        raise Exception(f'<which_pkflux_noise> not recognized: {which_pkflux_noise}')

    # Calculate the peak flux SNR
    #linedata.snr_pkflux = linedata.pkflux / linedata.pkfluxerr_for_snr
    linedata.snr_pkflux = linedata.pkflux_obs / linedata.pkfluxerr_for_snr

    # == (SNR) peak flux ==
    # Determine the method for calculating the flux noise
    if which_flux_noise == 'lmfit':
        linedata.fluxerr_for_snr = linedata.fluxerr

    else:
        raise Exception(f'<which_flux_noise> not recognized: {which_flux_noise}')

    # Calculate the flux SNR
    linedata.snr_flux = linedata.flux / linedata.fluxerr_for_snr

    # List of keys to be masked
    keylist = ['flux', 'fluxerr', 'sig', 'sigerr', 'wave', 'waveerr',
               'vel', 'velerr', 'pkflux', 'pkfluxerr', 'snr_pkflux', 'snr_flux',
               'pkfluxerr_for_snr', 
               'pkflux_obs', 'sig_obs', 'pkfluxerr_obs', 'sigerr_obs',
               ]

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

    if 'indices_ordering' in linedat.data:
        linedata.indices_ordering = linedat.data['indices_ordering'] * u.Unit(1)

        keylist += ['indices_ordering']

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

        if velmin is not None:
            mask_velmin = linedata.vel < velmin
            mask_bad = (mask_bad.T + mask_velmin.T).T

        if velmax is not None:
            mask_velmax = linedata.vel > velmax
            mask_bad = (mask_bad.T + mask_velmax.T).T

        if redchisqmax is not None:
            mask_redchi_sq = linedata.redchisq > redchisqmax
            mask_bad = (mask_bad.T + mask_redchi_sq.T).T

        if SNR_cut is not None:

            if SNR_cut_flux is None:
                SNR_cut_flux = SNR_cut

            mask_snr_flux = (linedata.snr_flux < SNR_cut_flux) +\
                                (~np.isfinite(linedata.snr_flux))

            mask_snr_pkflux = (linedata.snr_pkflux < SNR_cut) +\
                                (~np.isfinite(linedata.snr_pkflux))

            mask_snr_pkflux_range = ((linedata.snr_pkflux < SNR_cut) &\
                                        (linedata.snr_pkflux > SNR_cut_low)) +\
                                        (~np.isfinite(linedata.snr_pkflux))

            # Determine the method for selecting the SNR
            if which_snr_selection == 'flux or pkflux':
                #linedata.snr = linedata.snr_pkflux | linedata.snr_flux
                mask_snr = mask_snr_flux | mask_snr_pkflux

            elif which_snr_selection == 'flux if pkflux is low range':

                mask_snr = mask_snr_pkflux + False
                mask_snr[mask_snr_pkflux_range] = mask_snr_flux[mask_snr_pkflux_range]


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
                if mask3D.shape == data.shape:
                    data_masked = np.ma.array(data, mask=mask3D)
                else:
                    #mask3D_x = 
                    if key == 'indices_ordering':
                        data_masked = np.ma.array(data, mask=data==-99)

                    #print(key, mask3D.shape, data.shape)
            elif data.ndim == 2:
                data_masked = np.ma.array(data, mask=mask2D)
            else:
                raise Exception(f"<ndim> = {ndim} not implemented.")

            setattr(linedata, key, data_masked)

        linedata.mask3D = mask3D
        linedata.mask2D = mask2D
        keylist_masks = ['mask2D', 'mask3D']

        linedata.mask_snr_flux = mask_snr_flux
        linedata.mask_snr_pkflux = mask_snr_pkflux
        linedata.mask_snr = mask_snr

        keylist_masks += ['mask_snr_flux', 'mask_snr_pkflux', 'mask_snr']

        keylist += keylist_masks



    # Transpose
    if transpose:
        for key in keylist:
            setattr(linedata, key, getattr(linedata, key).T)

    #linedata.linedat = linedat
    return linedata


class ContData:
    def __init__(self, q3di):

        if not isinstance(q3di, q3din):
            q3di = q3dutil.get_q3dio(q3di)

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

def LineData_get_errors(self, linedata, line):

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

# ============= I/O ===================================
def read_quantity(hdu, ext='all', squeeze=True, unit=None, spectral=False):
	             #, wcs=False):
    '''
    hdu: filename or hdu
    ext: 'all', extension number or name.
    '''

    # load hdu
    if isinstance(hdu, str):
        hdu = fits.open(hdu)

    assert isinstance(hdu, fits.hdu.hdulist.HDUList)

    # ext
    if ext == 'all':
        ext = list(np.arange(len(hdu)))

        if len(hdu[ext[0]].shape) == 0:
            ext.remove(ext[0])
    
    if not is_iterable(ext):
        ext1 = [ext]
    else:
        ext1 = ext
    
    data1 = []
    spectral1 = []


    for iext in ext1:
        try:
            unit1 = u.Unit(hdu[iext].header['bunit'])
        except:
            if unit is None:
                unit1 = u.Unit('')
            else:
                unit1 = unit

        assert hdu[iext].data is not None
        data = hdu[iext].data * unit1

        if squeeze:
            data = np.squeeze(data)
        
        if spectral:
            spectral1 += [get_spectral(hdu[iext].header)]

        data1 += [data]


    if len(data1) == 1:
        data1 = data1[0]

    if spectral:
        return spectral1, data1
    else:
        return data1

def get_wcs(he, ext=0, wcs_type='all'):
    '''
    he: header, filename or hdu
    ext: extension number or name
    wcs_type: "all", "spectral" or "spatial"
    '''
    if isinstance(he, WCS):
        return he
    else:
        if not isinstance(he, fits.header.Header):
            if isinstance(he, str):
                he = fits.getheader(he, ext)
            elif isinstance(he, fits.hdu.hdulist.HDUList):
                he = he[ext].header
            else:
                raise Exception("Unexpected <he> input type:", type(he))

        wcs = WCS(he)

        wcs = get_wcs_type(wcs, wcs_type)

        return wcs
    
def get_wcs_type(wcs, wcs_type='all'):
    wcs_type = wcs_type.lower()
    if wcs_type in ['celestial', 'spatial']:
        wcs = wcs.celestial
    elif wcs_type in ['spectral']:
        wcs = wcs.spectral
    else:
        assert wcs_type == 'all', "Unexpected <wcs_type>: "+wcs_type
    
    return wcs

def get_spectral(header, unit_out=None):
    '''
    Input
    -----
    header: astropy.io.fits.Header or astropy.wcs.WCS
        header or wcs

    Output
    ------
    spectral: astropy.units.Quantity
        Spectral axis, obtained from the header
    '''
    # get wcs (spectral axis)
    wcs = get_wcs(header)
    wcs_spectral = wcs.spectral

    # get_unit
    s_unit = u.Unit(wcs_spectral.world_axis_units[0])
    if unit_out is not None:
        unit_out = u.Unit(unit_out)
    else:
        #unit_out = u.Unit(header['cunit3'])
        #unit_out = u.Unit(wcs.to_header()['cunit1'])
        unit_out = u.Unit(header[f'CUNIT{wcs.wcs.spec + 1}'])

    # spectral
    spectral = wcs_spectral.all_pix2world(np.arange(wcs_spectral.array_shape[0]), 0)[0]
    spectral *= s_unit
    
    spectral = spectral.to(unit_out)

    return spectral

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
def subtract_continuum(q3di, out_cubefile=None, overwrite=False, save_dq=True,
                       filename_dq=None, degree_polynomial=None):

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

    if degree_polynomial is not None:
        hdu[0].header['HISTORY'] = f"Continuum subtracted cube. Continuum obtained from a  {degree_polynomial}-degree polynomial fit ({np.datetime64('today', 'D')}, by Dall'Agnol de Oliveira, B.)." 
    hdu.writeto(out_cubefile, overwrite=overwrite)

    if save_dq:
        if filename_dq is None:
            filename_dq = '{}/aux/{}'.format(os.path.dirname(q3di.infile),
                                             os.path.basename(q3di.infile).replace('.fits', '_dq-original.fits'))

        data_dq = hdu['DQ'].data.astype(int)
        header_dq = hdu['DQ'].header
        #header = WCS(header).celestial.to_header()
        #dq_original.T.astype(bool)

        write_simplefits(data_dq, filename_dq, header=header_dq,
                                overwrite=overwrite)


def save_initial_conditions(q3di, line='[OIII]5007', filename_guess=None,
                            niter_extend=3, smooth_sigma1=3, smooth_sigma2=0.5,
                            smooth_sigma_final=1, plot=True, opts_load_maps={},
                            fluxpk_noise_fitsfile=None, SNR_cut=3,
                            writefitsfile=True):
    '''
    
    '''


    if not isinstance(q3di, q3din):
        
        q3di = q3dutil.get_q3dio(q3di)

    # load cube
    cube = q3di.load_cube()

    # load linemaps
    opts_load_maps1 = dict(cube=cube, transpose=True, apply_mask=True,
                          SNR_cut=SNR_cut, 
                          fluxpk_noise_fitsfile=fluxpk_noise_fitsfile)
    
    opts_load_maps1.update(opts_load_maps)
    assert opts_load_maps1['fluxpk_noise_fitsfile'] is not None, "Inform noise"

    linemaps = load_linemaps(q3di, line, **opts_load_maps1) 

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
    if niter_extend > 0:
        for parinit in (zinit, peakinit, siginit):

            convolve_opts = dict(boundary='extend')

            # loop over parameters
            for pari in parinit:

                # Smooth (extend region fittef)
                # Every iteration growths the region with initial values data
                for _ in range(niter_extend):
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
    if not os.path.exists(outdir_guess):
        os.makedirs(outdir_guess)

    if filename_guess is None:
        filename_guess = f'{outdir_guess}/init_{q3di.label}.fits'

    data = [peakinit, zinit, siginit]

    write_multifits(data, filename_guess,
                     header=None, name=['peakinit', 'zinit', 'siginit'],
                     overwrite=True)
    return data


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
    #from copy import deepcopy

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
    #print(f'do_base_fit = {do_base_fit}')

    if do_base_fit:
        #print(f'do_base_fit = {do_base_fit}')
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
    #if 0:
    #    if filename_q3di is not None:
    #        q3di_run = filename_q3di
    #    else:
    #        q3di_run = q3di

    #else:
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
                #print("Selecting the 10 closest good pixels")
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

    return spaxels_to_iter2D.T


# =====================================
def get_mask_fitted_spaxels(q3di,  
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

    mask_fitted_spaxels2D = np.zeros((q3di1.ncols, q3di1.nrows), dtype=bool)
    #k = 0
    for xcol, yrow in tqdm(zip(xcolarr, yrowarr), total=nspax,
                               desc='(get fitted spaxels) Spaxel: '):
    #for xcol, yrow in zip(xcolarr, yrowarr):
        col, row = xcol + 1, yrow + 1

        if os.path.isfile(get_filelab(q3di1, col, row)):
            #print(k)

            #q3do = load_q3dout(q3di1, col, row)

            # Test error
            #bool_error = get_bool_error(q3do, type_error)

            mask_fitted_spaxels2D[xcol, yrow] = True

    return mask_fitted_spaxels2D.T

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
                     cube=None, ncores=1,
                     filename_dq_original=None,
                     filename_dq_masked=None):
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
    # if called inside spiral_loop (individual spaxel)
    single_spaxel = False
    if isinstance(cols, (int, np.int64)) and isinstance(rows, (int, np.int64)):
        nspax, xcolarr, yrowarr = 1, np.array([cols-1]), np.array([rows-1])
        single_spaxel = True 
        #print('single_spaxel = ', single_spaxel)
    # if plotting a bunch of spaxels
    else:
        spaxels_to_iter2D = get_bad_spaxels_to_iter(q3di1, type_error, 
                                                    xcolarr=None, yrowarr=None, 
                                                    cols=cols, rows=rows,
                                                    cube=cube)

        # Update dq
        cube.dq[~spaxels_to_iter2D.T] = 1

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

            if not single_spaxel:            
                # Get remaining spaxels to fit and update cube.dq
                spaxels_to_iter2D = get_bad_spaxels_to_iter(q3di_object, type_error, 
                                                    xcolarr=None, yrowarr=None, 
                                                    cols=cols, rows=rows,
                                                    cube=cube)
                # update cube.dq
                cube.dq[~spaxels_to_iter2D.T] = 1
                # save updated dq extension
                #cube.writefits(q3di_object.infile)
                #print('1 single_spaxel = ', single_spaxel)

                hdu = fits.open(cube.infile)
                hdu['DQ'].data = cube.dq.T.astype(int)
                hdu.writeto(cube.infile, overwrite=True)

                # new number of spaxels to fit
                nspax_to_iter = np.sum(spaxels_to_iter2D)

    # Reset DQ mask
    if not single_spaxel:

        cube.dq = dq_original_module
        # save updated dq extension
        #cube.writefits(q3di1.infile)
        #print('2 single_spaxel = ', single_spaxel)

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
    if is_iterable(rows):
        row_slice = slice(rows[0]-1, rows[1])
    else:
        row_slice = slice(rows-1, rows)

    if is_iterable(cols):
        col_slice = slice(cols[0]-1, cols[1])
    else:
        col_slice = slice(cols-1, cols)

    #print(222222222222, row_slice, col_slice)

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

