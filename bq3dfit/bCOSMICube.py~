# %%
import os.path
import numpy as np
import matplotlib.pyplot as plt
import json

# external
from astropy.io import fits
from astropy import units as u
import pandas 

# q3dfit
from q3dfit import q3dutil
from q3dfit.q3din import q3din
from q3dfit.q3df import q3dfit
from q3dfit.q3dcollect import q3dcollect
from q3dfit.q3dout import load_q3dout
import q3dfit.q3dpro as q3dpro

# bruno
from bcube import ctools, plots, cplots, bcube, utils
from bq3dfit import bq3dutils

# %%
from copy import deepcopy
#linedat1 = deepcopy(linedat)

def rebuild_npz(filename_npz, filename_npz_red, save_index_ordering=True):

    linedat = np.load(filename_npz, allow_pickle=True)
    linedat_red = np.load(filename_npz_red, allow_pickle=True)


    linedat1 = {}

    ## %%% ------ 2D arrays ------
    for key in ['ndata', 'dof',  'redchisq', 'maxncomp']:
        linedat1[key] = deepcopy(linedat[key])

    ## %%% ------ 1D arrays ------
    for key in ['ncols', 'nrows']:
        linedat1[key] = deepcopy(linedat[key])

    ## %% --- 4D arrays --- 
    keylist4D = ['emlflx', 'emlflxerr', 'emlncomp', 'emlsig',
                 'emlsigerr', 'emlwav', 'emlwaverr', 'emlweq']
    for key in keylist4D:
        linedat1[key] = deepcopy(linedat_red[key])

    if save_index_ordering:
        indices = get_ordering(filename_npz, filename_npz_red)
        linedat1['indices_ordering'] = indices

    np.savez(filename_npz_red, **linedat1)


def get_ordering(filename_npz, filename_npz_red):


    linedat = np.load(filename_npz, allow_pickle=True)
    linedat_red = np.load(filename_npz_red, allow_pickle=True)

    maxncomp = len([i for i in linedat['emlweq'].item().keys() if 'fc' in i])
    maxncomp_red = len([i for i in linedat_red['emlweq'].item().keys() if 'fc' in i])

    #keylist = ['emlflx', 'emlflxerr', 'emlsig', 'emlsigerr', 'emlwav', 'emlwaverr']#, 'emlncomp', 'emlweq']
    keylist = ['emlflx', 'emlsig', 'emlwav']#, 'emlncomp', 'emlweq']

    #lines = q3di.lines
    lines = [line for line in linedat['emlflx'].item()['ftot'].keys() if '+' not in line]

    linemaps = {}
    linemaps_red = {}

    ncols = linedat['ncols']
    nrows = linedat['nrows']

    for line in lines:

        pars = {}
        pars_red = {}

        for par_key in keylist:    

            data_key_item = linedat[par_key].item()
            data_key_item_red = linedat_red[par_key].item()

            pars[par_key] = np.ma.zeros((ncols, nrows, maxncomp))
            pars_red[par_key] = np.ma.zeros((ncols, nrows, maxncomp_red))

            for comp in range(maxncomp):
                comp_key_match_list = [ikey for ikey in data_key_item.keys() if f'c{comp+1}' in ikey]

                for comp_key_match in comp_key_match_list:
                    pars[par_key][:,:,comp] =  data_key_item[comp_key_match][line]

            for comp in range(maxncomp_red):
                comp_key_match_list_red = [ikey for ikey in data_key_item_red.keys() if f'c{comp+1}' in ikey]

                for comp_key_match in comp_key_match_list_red:
                    pars_red[par_key][:,:,comp] = data_key_item_red[comp_key_match][line]

            pars[par_key].mask = np.isnan(pars[par_key])
            pars_red[par_key].mask = np.isnan(pars_red[par_key])

        linemaps[line] = pars
        linemaps_red[line] = pars_red


    # %%
    line = lines[-1]

    indices_par = {}
    for par_key in keylist:

        indices_par_line = np.zeros_like(linemaps[line][par_key], dtype=int)
        indices_par_line_T = indices_par_line.T

        par_line_T = linemaps[line][par_key].T
        par_line_red_T = linemaps_red[line][par_key].T

        for x in range(ncols):  # Iterate over columns
            for y in range(nrows):  # Iterate over rows
                #print(x, y)

                # old
                #indices_par_line_T[:, y, x] = [np.where(par_line_T[k, y, x] == par_line_red_T[:, y, x])[0][0] \
                #                                    for k in range(maxncomp)]

                # inverted
                indices_par_line_T[:, y, x] = [np.where(par_line_red_T[k, y, x] == par_line_T[:, y, x])[0][0] \
                                                    for k in range(maxncomp)]



        indices_par[par_key] = indices_par_line_T.T

    # check if indices are equal
    for par_key in keylist[1:]:

        assert (indices_par[keylist[0]] == indices_par[par_key]).all()
        assert (indices_par[keylist[0]] == indices_par[par_key]).all()

    indices = indices_par[keylist[0]]
    indices[linemaps[line][keylist[0]].mask] = -99

    #mask_indices = indices.mask
    #indices = indices.data.astype(float)
    #indices[mask] = np.nan

    # check plot
    if 0:
        # %%
        par_key = 'emlsig'
        line = lines[-1]
        sig = linemaps[line][par_key]
        sig_red = linemaps_red[lines[-1]][par_key]


        sig1 = np.zeros((ncols, nrows, maxncomp_red)) * np.nan
        sig1[:,:,:maxncomp] = sig

        indices1 = np.zeros((ncols, nrows, maxncomp_red), dtype=int)# * np.nan
        indices1[:,:,:maxncomp] = indices

        #indices1 = np.ma.array(indices1, mask=np.isnan(indices1))
        #indices1[:,:,maxncomp:] = True

        sig1 = np.take_along_axis(sig1, indices1, axis=-1)


        #plt.close('all')
        plt.imshow(sig[:,:,0].T, origin='lower')
        plt.show()

        #plt.close('all')
        plt.imshow(sig_red[:,:,0].T, origin='lower')
        plt.show()

        #plt.close('all')
        plt.imshow(sig1[:,:,0].T, origin='lower')
        plt.show()
        #plt.imshow(np.ma.array(sig1[:,:,0], mask=sig_red[:,:,0].mask).T, origin='lower')

    return indices








def plot_maps_line(filename_q3di, cube=None, filename_npz=None, output_dir=None,
                   opts_load_maps={}, n_gauss=None, pdffile=None, plot_opts_vel={}):

    homedir = utils.get_homedir()
    q3di = q3dutil.get_q3dio(filename_q3di)

    if cube is None:
        cube = q3di.load_cube()

    opts_load_maps['cube'] = cube 
    #opts_load_maps['transpose'] = False                       

    if 'merge_master' in q3di.outdir:
        opts_load_maps['type_mask2D'] = 'all'                        

    opts_load_maps['filename_npz'] = filename_npz
    #if opts_load_maps['filename_npz'] is not None:
    #    opts_load_maps['redchisqmax'] = None
    #    opts_load_maps['stats'] = False
    #    #opts_load_maps['stats'] = False


    if 1:
        line = '[OIII]5007'
        linemaps_oIII = bq3dutils.load_linemaps(q3di, line, **opts_load_maps) 
        linemaps_oIII.label = '[OIII]$\,$5007'

    line = 'Hbeta'
    linemaps_hbeta = bq3dutils.load_linemaps(q3di, line, **opts_load_maps) 

    if 0:
        linemaps_all = bq3dutils.load_linemaps_all(q3di, **opts_load_maps) 

        line = '[OIII]5007'
        linemaps_oIII = linemaps_all[line]
        linemaps_oIII.label = '[OIII]$\,$5007'

    # Cont
    if 1:
        if 0:
            contdata = bq3dutils.ContData(q3dutil.get_q3dio(bprojects.get_filename_q3di(
                                                            indir, shortname, region_fit,
                                                            n_gauss=1, forcont=True)))

            img_cont = np.nanmean(contdata.all_mod.T, axis=0)
        else:
            filename_cont = homedir + '/data/HzRGs_Sample/4C+03.24/data_other_wavelengths/4C+03.24_continuum_wuji.fits'
            img_cont = fits.getdata(filename_cont)
            img_cont[cube.dq.T.astype(bool).all(axis=0)] = np.nan

    # red chi
    #if opts_load_maps['filename_npz'] is None:
    if 1:
        redchisq = linemaps_oIII.redchisq
        maxncomp = linemaps_oIII.maxncomp

    # WCS
    from astropy import coordinates
    coo0 = coordinates.SkyCoord(
                                ra='12:45:38.377', dec='+03:23:21.14',
                                # ra='12:45:38.37', dec='+03:23:20.98',
                                unit=(u.hourangle, u.deg))
    #ra0, dec0 = coo0.ra.degree, coo0.dec.degree
    ra0, dec0 = coo0.ra, coo0.dec

    dra, ddec = ctools.get_wcs_map(cube.header_dat, radec0=(ra0, dec0))

    # Radio
    if 1:
        filename_radio = homedir + '/data/HzRGs_Sample/4C+03.24/data_other_wavelengths/4C03.24_rob0_VLA.fits'
        img_radio = fits.getdata(filename_radio)[0,0]
        he_radio = fits.getheader(filename_radio)

        dra_radio, ddec_radio = ctools.get_wcs_map(he_radio,
                                    #radec0=(191.4099124*u.deg,3.3892069*u.deg))
                                    radec0=(ra0, dec0))

    # ============================== PLOT ================== 
    # size

    if n_gauss is None:
        if filename_npz is not None:
            linedat = np.load(filename_npz, allow_pickle=True)
            n_gauss = len([i for i in linedat['emlweq'].item().keys() if 'fc' in i])
        else:
            n_gauss = len(linemaps_oIII.flux)


    #nLins = n_gauss + 2
    #n_gauss = 5
    nLins = n_gauss + 1
    

    if 0:
        S = 1
        nCols = 3

    if 1:

        S = 2
        nCols = 3 + 1 + 2

    figsize = np.array([6, 5.4*(nLins/nCols)]) * S

    # opts
    title_opts = {'y':1.25}

    # plot
    plt.close('all')
    fig, ax = plt.subplots(nLins, nCols, figsize=figsize)
    ax = ax.reshape(nLins, nCols)

    for idx in range(n_gauss):
        # Flux
        img = linemaps_oIII.flux[idx]
        label_unit = '$\\rm{{erg\,s^{-1}\,cm^{-2}}}$'
        label = f'FLUX [OIII]_c{idx+1}\n({label_unit})'
        vpercent=[1, 99]
        plots.plot_quantity_map(img, axis=ax[idx,0],  doLog=True, 
                                vpercent=vpercent,
                                X=dra,Y=ddec,
                                do_contour=False,
                                contour_opts={'nlevels':3},
                                do_colorbar=True,
                                divergence=False, cmap='inferno',
                                title=label, title_opts=title_opts)

        # Radial velocity
        img = linemaps_oIII.vel[idx]
        label_unit = '$\\rm{{km\,s^{-1}}}$'
        label = f'VEL [OIII]_c{idx+1}\n({label_unit})'
        vpercent = [8, 92] 
        plots.plot_quantity_map(img, axis=ax[idx,1],  doLog=False, 
                                X=dra,Y=ddec,
                                #vpercent=vpercent,
                                vpercent=None, vmin=-1500, vmax=1500,
                                do_contour=False,
                                contour_opts={'nlevels':3},
                                do_colorbar=True,
                                divergence=True, cmap='coolwarm',
                                title=label, title_opts=title_opts)
        if 1:
            # Radial velocity
            img = linemaps_oIII.vel[idx]
            label_unit = '$\\rm{{km\,s^{-1}}}$'
            label = f'VEL [OIII]_c{idx+1}\n({label_unit})'
            vpercent = [3, 97] 
            
            plot_opts1 = dict(vpercent=vpercent,
                                    #vpercent=None, vmin=-1500, vmax=1500,
                                    do_contour=False,
                                    contour_opts={'nlevels':3},
                                    do_colorbar=True,
                                    divergence=False, cmap='coolwarm',
                                    title=label, title_opts=title_opts)
            plot_opts1.update(plot_opts_vel)
            plots.plot_quantity_map(img, axis=ax[idx,4],  doLog=False, 
                                    X=dra,Y=ddec, **plot_opts1)



        # Velocity dispersion
        img = linemaps_oIII.sig[idx]
        label_unit = '$\\rm{{km\,s^{-1}}}$'
        label = f'SIGMA [OIII]_c{idx+1}\n({label_unit})'
        vpercent = [8, 95] 
        out = plots.plot_quantity_map(img, axis=ax[idx,2],  doLog=True, 
                                X=dra,Y=ddec,
                                #vpercent=vpercent,
                                vpercent=None, vmin=30, vmax=1000,
                                do_contour=False,
                                do_colorbar=True,
                                contour_opts={'nlevels':3},
                                divergence=False, cmap='magma',
                                title=label, title_opts=title_opts)
        ticks_minor = out['cbar'].ax.get_yticks(minor=True)
        ticks_major = out['cbar'].ax.get_yticks(minor=False)
        ticklabels_minor = np.array([f'{tick:.5g}' for tick in ticks_minor])
        ticklabels_major = np.array([f'{tick:.5g}' for tick in ticks_major])
        #ticklabels_minor[1::3] = ''
        #ticklabels_minor[2::3] = ''
        #ticklabels_minor[1::2] = ''
        ticklabels_minor = [tick if tick[0] not in ['9', '8', '7', '5', '4', '2'] else '' for tick in ticklabels_minor ]
        out['cbar'].ax.set_yticklabels(ticklabels_minor, minor=True)
        out['cbar'].ax.set_yticklabels(ticklabels_major, minor=False)
        out['cbar'].ax.minorticks_on()

        if 1:
            # Velocity dispersion
            img = linemaps_oIII.sig[idx]
            label_unit = '$\\rm{{km\,s^{-1}}}$'
            label = f'SIGMA [OIII]_c{idx+1}\n({label_unit})'
            vpercent = [5, 95] 
            out = plots.plot_quantity_map(img, axis=ax[idx,5],  doLog=True, 
                                    X=dra,Y=ddec,
                                    vpercent=vpercent,
                                    #vpercent=None, vmin=30, vmax=1000,
                                    do_contour=False,
                                    do_colorbar=True,
                                    contour_opts={'nlevels':3},
                                    divergence=False, cmap='magma',
                                    title=label, title_opts=title_opts)
            ticks_minor = out['cbar'].ax.get_yticks(minor=True)
            ticks_major = out['cbar'].ax.get_yticks(minor=False)
            ticklabels_minor = np.array([f'{tick:.5g}' for tick in ticks_minor])
            ticklabels_major = np.array([f'{tick:.5g}' for tick in ticks_major])
            #ticklabels_minor[1::3] = ''
            #ticklabels_minor[2::3] = ''
            #ticklabels_minor[1::2] = ''
            ticklabels_minor = [tick if tick[0] not in ['9', '8', '7', '5', '4', '2'] else '' for tick in ticklabels_minor ]
            out['cbar'].ax.set_yticklabels(ticklabels_minor, minor=True)
            out['cbar'].ax.set_yticklabels(ticklabels_major, minor=False)
            out['cbar'].ax.minorticks_on()

        # [OIII] / Hbeta
        #oiii_hbeta = linemaps_oIII.flux / linemaps_hbetadlux
        if 1:
            img = linemaps_oIII.flux[idx] / linemaps_hbeta.flux[idx]
            label = f'([OIII]/H$\\beta$)_c{idx+1}'
            vpercent = 5#[2, 98]
            doLog = False
            out = plots.plot_quantity_map(img, axis=ax[idx, 3],  doLog=doLog, 
                                    X=dra,Y=ddec,
                                    #vpercent=vpercent,
                                    vmin=1, vmax=13,
                                    do_contour=True,
                                    do_colorbar=True,
                                    contour_opts=dict(nlevels=4),
                                    divergence=False, cmap='cividis',
                                    title=label, title_opts=title_opts)

            if doLog:
                ticks_minor = out['cbar'].ax.get_yticks(minor=True)
                ticks_major = out['cbar'].ax.get_yticks(minor=False)
                ticklabels_minor = np.array([f'{tick:.5g}' for tick in ticks_minor])
                ticklabels_major = np.array([f'{tick:.5g}' for tick in ticks_major])
                ticklabels_minor[0::3] = ''
                ticklabels_minor[1::3] = ''
                #ticklabels_minor[1::2] = ''
                out['cbar'].ax.set_yticklabels(ticklabels_minor, minor=True)
                out['cbar'].ax.set_yticklabels(ticklabels_major, minor=False)
                out['cbar'].ax.minorticks_on()








    flux_tot_oIII = np.sum(linemaps_oIII.flux, axis=0)

    
    # [OIII]
    img = flux_tot_oIII
    label_unit = '$\\rm{{erg\,s^{-1}\,cm^{-2}}}$'
    label = f'FLUX [OIII]_tot\n({label_unit})'
    vpercent=[1, 99]
    plots.plot_quantity_map(img, axis=ax[-1,0],  doLog=True, 
                                X=dra,Y=ddec,
                            vpercent=vpercent,
                            do_contour=True,
                            do_colorbar=True,
                            divergence=False, cmap='inferno',
                            title=label, title_opts=title_opts)

    flux_tot_hbeta = np.sum(linemaps_hbeta.flux, axis=0)
    if 1:

        # Hbeta
        if not flux_tot_hbeta.mask.all():
            #img = flux_tot_hbeta
            #label_unit = '$\\rm{{erg\,s^{-1}\,cm^{-2}}}$'
            #label = f'FLUX (Hbeta)_tot\n({label_unit})'
            #vpercent=[1, 99]

            #plots.plot_quantity_map(img, axis=ax[-1,1],  doLog=True, 
            #                        X=dra,Y=ddec,
            #                        vpercent=vpercent,
            #                        do_contour=True,
            #                        do_colorbar=True,
            #                        divergence=False, cmap='inferno',
            #                        title=label, title_opts=title_opts)

            # [OIII] / Hbeta
            img = flux_tot_oIII / flux_tot_hbeta
            # label_unit = '$\\rm{{erg\,s^{-1}\,cm^{-2}}}$'
            label = f'([OIII]/H$\\beta$)_tot'# ({label_unit})'
            vpercent=5#[2, 98]
            doLog = False
            out = plots.plot_quantity_map(img, axis=ax[-1,3],  doLog=doLog, 
                                    X=dra,Y=ddec,
                                    #vpercent=vpercent,
                                    vmin=1, vmax=13,
                                    do_contour=True,
                                    do_colorbar=True,
                                    contour_opts=dict(nlevels=4),
                                    divergence=False, cmap='cividis',
                                    title=label, title_opts=title_opts)

            if doLog:
                ticks_minor = out['cbar'].ax.get_yticks(minor=True)
                ticks_major = out['cbar'].ax.get_yticks(minor=False)
                ticklabels_minor = np.array([f'{tick:.5g}' for tick in ticks_minor])
                ticklabels_major = np.array([f'{tick:.5g}' for tick in ticks_major])
                ticklabels_minor[0::3] = ''
                ticklabels_minor[1::3] = ''
                #ticklabels_minor[1::2] = ''
                out['cbar'].ax.set_yticklabels(ticklabels_minor, minor=True)
                out['cbar'].ax.set_yticklabels(ticklabels_major, minor=False)
                out['cbar'].ax.minorticks_on()

    # ========================
    if 1:#opts_load_maps['filename_npz'] is None:

        for axis  in ax[:, 0].ravel():
            # Continuum - Contour
            title = None#'Optical cont. /\n radio cont. (15 GHz)'
            plots.plot_quantity_map(img_cont, axis=axis,  doLog=True, 
                                        X=dra,Y=ddec,
                                            vmin=400, vmax=None, vpercent=[1,100],
                                            do_contour=True,
                                            do_image=False,
                                            contour_opts=dict(nlevels=3, 
                                                              colors='grey', 
                                                              linestyles='-',
                                                              linewidths=1, alpha=0.75),
                                            do_colorbar=False,
                                            #title=title, title_opts=title_opts
                                            )


    if 1:
        # Continuum - radio
        #for axis in [ax[-2,0], ax[-2,1], ax[-2,2]]:
        #for axis in ax[:,0].ravel():
        for axis in ax.ravel():

            #contour_opts=dict(
            #                vlim=[3*4.33313e-06, None],
            #                nlevels=4, colors='red',linewidths=1.8, alpha=0.5)

            contour_opts=dict(
                            vlim=[3.5*4.33313e-06, None],
                            nlevels=3, colors='blue', linewidths=1., alpha=0.5)


            plots.plot_quantity_map(img_radio, axis=axis,  doLog=True, 
                                            X=dra_radio, Y=ddec_radio,
                                            vpercent=(0.01, 99.9),
                                            #vpercent=(0.05, 99.9),
                                            do_contour=True,
                                            do_image=False,
                                            contour_opts=contour_opts,
                                            do_colorbar=False,
                                            )
                

    # ========================
    #if opts_load_maps['filename_npz'] is None:
    if 1:

        # Red chi
        #title = 'Red. $\chi_2$ /\n radio cont. (15 GHz)'
        title = 'Red. $\chi_2$'
        plots.plot_quantity_map(redchisq, axis=ax[-1,1],  doLog=False, 
                                    X=dra,Y=ddec,
                                        vmin=0.4, vmax=1.6,
                                        #vpercent=(0.5, 99.5),
                                        do_contour=False,
                                        do_image=True,
                                        cmap='berlin',
                                        #contour_opts=dict(nlevels=10, colors='black'),
                                        do_colorbar=True,
                                        title=title, title_opts=title_opts
                                        )


        # ========================
        # N GAUSS
        #title = '# Gausssian comp. /\n radio cont. (15 GHz)'
        title = '# Gausssian comp.'
        out = plots.plot_quantity_map(maxncomp, axis=ax[-1,2],  doLog=False, 
                                    X=dra,Y=ddec,
                                        #vpercent=(0.5, 99.5),
                                        do_contour=False,
                                        colorbar_opts=dict(ticks=np.arange(1, 7), 
                                                        boundaries=np.arange(0.5, 7.5, 1)),
                                        do_image=True,
                                        cmap='viridis',
                                        do_colorbar=True,
                                        title=title, title_opts=title_opts
                                        )

    #cbar0 = plt.colorbar(im0, ax=ax[0,0], ticks=np.arange(1, 7), boundaries=np.arange(0.5, 7.5, 1))

    # ========================
    # ========================

    for axis in ax.ravel():
        axis.set_xlim(2, -1)
        axis.set_ylim(-2.2, 1)

        #axis.set_xlabel("Delta RA (degree)")
        #axis.set_ylabel("Delta DEC (degree)")

        plots.set_ticks_locators(
                    axis,
                    x_MajMin=[1, 0.5],
                    y_MajMin=[1,0.5],
                )

        axis.scatter(0, 0, marker='x', color='black', s=5, lw=.5, zorder=10,)

        scatter_opts0_oIII = dict(marker='^', facecolors='None', edgecolors='blue',
                                   s=15, lw=.8, zorder=10,)
        scatter_opts0_cII = dict(marker='s', facecolors='None', edgecolors='red',
                                  s=15, lw=.8, zorder=10,)

        coo0 = coordinates.SkyCoord(ra='12:45:38.37', dec='+03:23:20.98',
                                    unit=(u.hourangle, u.deg))
        ra0, dec0 = coo0.ra.degree, coo0.dec.degree

        coos_oIII = []
        coos_cII = []
        coos_oIII += [coordinates.SkyCoord(ra='12:45:38.367', dec='+03:23:21.45',
                                            unit=(u.hourangle, u.deg))]
        coos_cII += [coordinates.SkyCoord(ra='12:45:38.377', dec='+03:23:20.96',
                                            unit=(u.hourangle, u.deg))]
        coos_cII += [coordinates.SkyCoord(ra='12:45:38.470', dec='+03:23:19.48',
                                            unit=(u.hourangle, u.deg))]
        coos_cII += [coordinates.SkyCoord(ra='12:45:38.420', dec='+03:23:21.00',
                                            unit=(u.hourangle, u.deg))]

        for cooi in coos_oIII:
            drai, ddeci = coo0.spherical_offsets_to(cooi)            
            axis.scatter(drai.to(u.arcsec).value, ddeci.to(u.arcsec).value,
                         **scatter_opts0_oIII)
                         #marker='^', facecolors='None', edgecolors='black', s=100, linewidths=2)

        for cooi in coos_cII:
            drai, ddeci = coo0.spherical_offsets_to(cooi)            
            axis.scatter(drai.to(u.arcsec).value, ddeci.to(u.arcsec).value,
                         **scatter_opts0_cII)

    # 
    plots.set_axis_labels_grid(ax.ravel(), nLins, nCols, inner_ticklabels='',
                            inner_axislabels='')

    # ========================
    fig.tight_layout()
    #fig.savefig(f'test_maps_ngauss{n_gauss}.png', bbox_inches='tight')
    #fig.savefig(f'figs/test_maps_ngauss{n_gauss}_SNR{SNR_cut}.pdf', bbox_inches='tight')

    if filename_npz is not None:
        label = os.path.basename(filename_npz).replace('.line', '').replace('.npz', '')
    else:
        label = q3di.label

    if output_dir is None:
        if filename_npz is not None:
            output_dir = os.path.dirname(filename_npz)
        else:
            output_dir = q3di.outdir

    pdffile = None
    if pdffile is None:
        pdffile = f'{output_dir}/maps_{label}.pdf'

    fig.savefig(pdffile, bbox_inches='tight')

    #fig.savefig(f'test_maps_ngauss{n_gauss}_SNR{SNR_cut}.pdf', bbox_inches='tight')

    # %%
    if 0:
        from astropy.wcs import WCS
        flux_tot_oIII
        header3D = cube.header_dat
        header2D = WCS(cube.header_dat).celestial.to_header()
        header2D['bunit'] = header3D['bunit']

        data = flux_tot_oIII.data.value
        bunit = flux_tot_oIII.data.unit
        data[flux_tot_oIII.mask] = np.nan
        data *= bunit

        ctools.write_simplefits(data,

            f'{os.path.dirname(q3di.infile)}/outfitmaps/{q3di.label}_fluxtot.fits',
            header=header2D, 
            overwrite=True, 
        )


#plot_maps_line(q3di, cube=cube, filename_npz=filename_npz)

















































