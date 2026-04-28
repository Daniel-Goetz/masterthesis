# -*- coding: utf-8 -*-
import copy as copy
import importlib.resources as pkg_resources
import matplotlib as mpl
import numpy as np
import os
import q3dfit.q3dutil as q3dutil

from astropy.table import Table
from q3dfit.data import linelists

from tqdm.auto import tqdm



def get_spaxels(cube, cols=None, rows=None, ignore_masked_spaxels=False):
    '''
    Set up 1D arrays specifying column value and row value at each point to be
    fitted. These are zero-offset for indexing other arrays.

    Parameters
    ----------
    cube : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is None.
    rows : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    # Set up 2-element arrays with starting and ending columns/rows
    # These are unity-offset to reflect pixel labels
    if not cols:
        cols = [1, cube.ncols]
        ncols = cube.ncols
        #   case: cols is a scalar
    elif not isinstance(cols, (list, np.ndarray)):
        cols = [cols, cols]
        ncols = 1
    elif len(cols) == 1:
        cols = [cols[0], cols[0]]
        ncols = 1
    else:
        ncols = cols[1]-cols[0]+1
    if not rows:
        rows = [1, cube.nrows]
        nrows = cube.nrows
    elif not isinstance(rows, (list, np.ndarray)):
        rows = [rows, rows]
        nrows = 1
    elif len(rows) == 1:
        rows = [rows[0], rows[0]]
        nrows = 1
    else:
        nrows = rows[1]-rows[0]+1

    if len(cols)<=2 or len(rows) <= 2:
        colarr = np.empty((ncols, nrows), dtype=np.int32)
        rowarr = np.empty((ncols, nrows), dtype=np.int32)
        for i in range(nrows):
            colarr[:, i] = np.arange(cols[0]-1, cols[1], dtype=np.int32)
        for i in range(ncols):
            rowarr[i, :] = np.arange(rows[0]-1, rows[1], dtype=np.int32)

        # Flatten from 2D to 1D arrays to preserve indexing using only ispax
        # currently not needed. fitloop expects 2D lists.
        colarr = colarr.flatten()
        rowarr = rowarr.flatten()

        # To speed up the fit, ignore spaxels totally flagged along the spectral
        # dimension. This is specially important in the MultCore fit, because
        # it avoid a given core looping over list full of 'bad' spaxels.
        if not ignore_masked_spaxels:
            nspax = ncols * nrows
        else:
            allgood = (~cube.dq.astype(bool).all(axis=-1))
            allgood = allgood[cols[0]-1:cols[1], rows[0]-1:rows[1]].flatten()

            colarr = colarr[allgood]
            rowarr = rowarr[allgood]
            nspax = len(colarr)

    if len(cols) > 2 or len(rows) > 2:
        colarr = cols
        rowarr = rows
        nspax = len(cols)

    return nspax, colarr, rowarr


def q3dcollect(q3di, cols=None, rows=None, quiet=True, compsortpar='sigma',
               compsortdir='up', ignoreres=False, ignore_masked_spaxels=False,
               sufix='', mask2D=None, mask3D=None):
    
    """
    Routine to collate spaxel information together.

    As input, it requires a q3din object and the q3dout objects output by
    the fit.

    Parameters
    ----------
    q3di: in, required, type=string
        Name of object to initialize the fit.
    cols: in, optional, type=intarr, default=all
        Columns to fit, in 1-offset format. Either a scalar or a
        two-element vector listing the first and last columns to fit.
    rows: in, optional, type=intarr, default=all
        Rows to fit, in 1-offset format. Either a scalar or a
        two-element vector listing the first and last rows to fit.
    quiet: in, optional, type=boolean
        Print error and progress messages. Propagates to most/all subroutines.
    compsortpar: in, optional, type=string, default='sigma'
        Parameter by which to sort components. Options are 'sigma', 'flux', 
        'wave'.
    compsortdir: in, optional, type=string, default='up'
        Direction in which to sort components. Options are 'up' and 'down'.
    ignoreres: in, optional, type=boolean, default=False
        Parameter passed to sepfitpars(). Ignore spectral resolution in 
        computing observed sigmas and peak  fluxes. This is mainly for 
        backward compatibility with old versions, which did not store the 
        spectral resolution in an easily accessible way in the specConv object.

    Returns
    -------
    None

    Raises
    ------
    None

    """

    #load initialization object
    q3di = q3dutil.get_q3dio(q3di)

    # set up linelist
    if q3di.dolinefit:

        print('Sorting components by '+compsortpar+' in the '+
              compsortdir+'ward direction.')

        linelist = q3dutil.get_linelist(q3di)

        # table with doublets to combine
        with pkg_resources.path(linelists, 'doublets.tbl') as p:
            doublets = Table.read(p, format='ipac')
        # make a copy of singlet list
        lines_with_doublets = copy.deepcopy(q3di.lines)
        # append doublet names to singlet list
        for (name1, name2) in zip(doublets['line1'], doublets['line2']):
            if name1 in linelist['name'] and name2 in linelist['name']:
                lines_with_doublets.append(name1+'+'+name2)

    # READ DATA
    cube, vormap = q3dutil.get_Cube(q3di, quiet=quiet)

    # process col, row specifications
    nspax, colarr, rowarr = get_spaxels(cube, cols=cols, rows=rows,
                                    ignore_masked_spaxels=ignore_masked_spaxels)
    # TODO
    # if q3di.vormap is not None:
    #     vormap = q3di.vromap
    #     nvorcols = max(vormap)
    #     vorcoords = np.zeros(nvorcols, 2)
    #     for i in range(0, nvorcols):
    #         xyvor = np.where(vormap == i).nonzero()
    #         vorcoords[:, i] = xyvor


    # Create output line dictionaries
    if q3di.dolinefit:
        emlwav = dict()
        emlwaverr = dict()
        emlsig = dict()
        emlsigerr = dict()
        emlweq = dict()
        emlflx = dict()
        emlflxerr = dict()
        emlncomp = dict()
        emlweq['ftot'] = dict()
        emlflx['ftot'] = dict()
        emlflxerr['ftot'] = dict()
        for k in range(0, q3di.maxncomp):
            cstr = 'c' + str(k + 1)
            emlwav[cstr] = dict()
            emlwaverr[cstr] = dict()
            emlsig[cstr] = dict()
            emlsigerr[cstr] = dict()
            emlweq['f' + cstr] = dict()
            emlflx['f' + cstr] = dict()
            emlflxerr['f' + cstr] = dict()
            emlflx['f' + cstr + 'pk'] = dict()
            emlflxerr['f' + cstr + 'pk'] = dict()
        for line in lines_with_doublets:
            emlncomp[line] = np.zeros((cube.ncols, cube.nrows), dtype=int)
            emlweq['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + np.nan
            emlflx['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                            dtype=float) + np.nan
            emlflxerr['ftot'][line] = np.zeros((cube.ncols, cube.nrows),
                                               dtype=float) + np.nan
            for k in range(0, q3di.maxncomp):
                cstr = 'c' + str(k + 1)
                emlwav[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                              dtype=float) + np.nan
                emlwaverr[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                 dtype=float) + np.nan
                emlsig[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                              dtype=float) + np.nan
                emlsigerr[cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                 dtype=float) + np.nan
                emlweq['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                  dtype=float) + np.nan
                emlflx['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                  dtype=float) + np.nan
                emlflxerr['f'+cstr][line] = np.zeros((cube.ncols, cube.nrows),
                                                     dtype=float) + np.nan
                emlflx['f'+cstr+'pk'][line] = \
                    np.zeros((cube.ncols, cube.nrows),
                             dtype=float) + np.nan
                emlflxerr['f'+cstr+'pk'][line] = \
                    np.zeros((cube.ncols, cube.nrows),
                             dtype=float) + np.nan

    # Stats parameters (BIC/AIC can be calculated from this later)
    redchisq = np.zeros((cube.ncols, cube.nrows), dtype=float) + np.nan
    dof = np.zeros((cube.ncols, cube.nrows), dtype=float) + np.nan
    ndata = np.zeros((cube.ncols, cube.nrows), dtype=float) + np.nan
    maxncomp = np.zeros((cube.ncols, cube.nrows), dtype=float) + np.nan

    # create output cubes
    if q3di.docontfit:
        hostcube = {'dat': np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'err': np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'dq':  np.zeros((cube.ncols, cube.nrows, cube.nwave)),
                    'norm_div': np.zeros((cube.ncols, cube.nrows,
                                          cube.nwave)),
                    'norm_sub': np.zeros((cube.ncols, cube.nrows,
                                          cube.nwave))}
        contcube = {'npts': np.zeros((cube.ncols, cube.nrows)) + np.nan,
                    'stel_rchisq': np.zeros((cube.ncols, cube.nrows)) + np.nan,
                    'stel_z': np.zeros((cube.ncols, cube.nrows)) + np.nan,
                    'stel_z_err': np.zeros((cube.ncols, cube.nrows, 2)) + np.nan,
                    'stel_ebv': np.zeros((cube.ncols, cube.nrows)) + np.nan,
                    'stel_ebv_err':
                        np.zeros((cube.ncols, cube.nrows, 2)) + np.nan,
                    'stel_sigma': np.zeros((cube.ncols, cube.nrows)) + np.nan,
                    'stel_sigma_err':
                        np.zeros((cube.ncols, cube.nrows, 2)) + np.nan}

        if q3di.decompose_ppxf_fit:
            contcube['all_mod'] = np.zeros((cube.ncols, cube.nrows,
                                            cube.nwave))
            contcube['stel_mod'] = np.zeros((cube.ncols, cube.nrows,
                                             cube.nwave))
            contcube['poly_mod'] = np.zeros((cube.ncols, cube.nrows,
                                             cube.nwave))
        elif q3di.decompose_qso_fit:
            contcube['qso_mod'] = np.zeros((cube.ncols, cube.nrows,
                                            cube.nwave))
            # contcube['qso_poly_mod'] = np.zeros((cube.ncols, cube.nrows,
            #                                      cube.nwave))
            contcube['host_mod'] = np.zeros((cube.ncols, cube.nrows,
                                             cube.nwave))
            contcube['poly_mod'] = np.zeros((cube.ncols, cube.nrows,
                                             cube.nwave))
            if q3di.decompose_qso_fit:
                contcube['all_mod'] = np.zeros((cube.ncols, cube.nrows,
                                            cube.nwave))
        else:
            contcube['all_mod'] = np.zeros((cube.ncols, cube.nrows,
                                            cube.nwave))

    # LOOP THROUGH SPAXELS

    # track first continuum fit
    firstcontfit = True

    for ispax in tqdm(range(0, nspax), total=nspax, desc='Spaxel: '):

        i = colarr[ispax]
        j = rowarr[ispax]

        if not quiet:
            print(f'Column {i+1} of {cube.ncols}')

        # set up labeling

        # set this to false unless we're using Voronoi binning
        # and the tiling is missing
        # vortile = True
        labin = '{0.outdir}{0.label}'.format(q3di)
        if cube.dat.ndim == 1:
            flux = cube.dat
            err = cube.err
            dq = cube.dq
            labout = labin
        elif cube.dat.ndim == 2:
            flux = cube.dat[:, i]
            err = cube.err[:, i]
            dq = cube.dq[:, i]
            labin += '_{:04d}'.format(i+1)
            labout = labin
        else:
            if not quiet:
                print(f'    Row {j+1} of {cube.nrows}')

            # TODO
            # if q3di.vormap is not None:
            #    if np.isfinite(q3di.vormap[i][j]) and \
            #            q3di.vormap[i][j] is not np.nan:
            #        iuse = vorcoords[q3di.vormap[i][j] - 1, 0]
            #        juse = vorcoords[q3di.vormap[i][j] - 1, 1]
            #    else:
            #        vortile = False
            #else:
            iuse = i
            juse = j

            #if vortile:
            flux = cube.dat[iuse, juse, :].flatten()
            err = cube.err[iuse, juse, :].flatten()
            dq = cube.dq[iuse, juse, :].flatten()
            labin = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
                format(q3di, iuse+1, juse+1)
            labout = '{0.outdir}{0.label}_{1:04d}_{2:04d}'.\
                format(q3di, i+1, j+1)

        # Restore fit after a couple of sanity checks
        # if vortile:
        infile = labin + '.npy'
        nodata = flux.nonzero()
        ct = len(nodata[0])
        # else:
        #    ct = 0

        if not os.path.isfile(infile):

            badmessage = f'        No data for [{i+1}, {j+1}]'
            print(badmessage)

        else:

            # load fit object
            q3do = q3dutil.get_q3dio(infile)

            # Restore original error.
            q3do.spec_err = err[q3do.fitran_indx]

            if q3do.dolinefit:

                # process line fit parameters
                q3do.sepfitpars(doublets=doublets, ignoreres=ignoreres)

                # get correct number of components in this spaxel
                thisncomp = 0
                thisncompline = ''

                for line in lines_with_doublets:
                    sigtmp = q3do.line_fitpars['sigma'][line]
                    fluxtmp = q3do.line_fitpars['flux'][line]
                    # TODO
                    igd = [idx for idx in range(len(sigtmp)) if
                           (sigtmp[idx] != 0 and
                            not np.isnan(sigtmp[idx]) and
                            fluxtmp[idx] != 0 and
                            not np.isnan(fluxtmp[idx]))]
                    ctgd = len(igd)

                    if ctgd > thisncomp:
                        thisncomp = ctgd
                        thisncompline = line

                    # assign total fluxes
                    if ctgd > 0:
                        emlflx['ftot'][line][i, j] = \
                            q3do.line_fitpars['tflux'][line]
                        emlflxerr['ftot'][line][i, j] = \
                            q3do.line_fitpars['tfluxerr'][line]

                    # assign to output dictionary
                    emlncomp[line][i, j] = ctgd

                if thisncomp == 1:
                    isort = [0]
                elif thisncomp >= 2:
                    igd = np.arange(thisncomp)
                    sortpars = q3do.line_fitpars[compsortpar][thisncompline]
                    isort = np.argsort(sortpars[igd])
                    if compsortdir == 'down':
                        isort = np.flip(isort)
                if thisncomp > 0:
                    for line in lines_with_doublets:
                        kcomp = 1
                        for sindex in isort:
                            cstr = 'c' + str(kcomp)
                            emlwav[cstr][line][i, j] \
                                = q3do.line_fitpars['wave'][line].data[sindex]
                            emlwaverr[cstr][line][i, j] \
                                = q3do.line_fitpars['waveerr'][line].data[sindex]
                            emlsig[cstr][line][i, j] \
                                = q3do.line_fitpars['sigma'][line].data[sindex]
                            emlsigerr[cstr][line][i, j] \
                                = q3do.line_fitpars['sigmaerr'][line].data[sindex]
#                            emlweq['f' + cstr][line][i, j] \
#                                = lineweqs['comp'][line].data[sindex]
                            emlflx['f' + cstr][line][i, j] \
                                = q3do.line_fitpars['flux'][line].data[sindex]
                            emlflxerr['f' + cstr][line][i, j] \
                                = q3do.line_fitpars['fluxerr'][line].data[sindex]
                            emlflx['f' + cstr + 'pk'][line][i, j] \
                                = q3do.line_fitpars['fluxpk'][line].data[sindex]
                            emlflxerr['f' + cstr + 'pk'][line][i, j] \
                                = q3do.line_fitpars['fluxpkerr'][line].data[sindex]
                            kcomp += 1

            redchisq[i, j] = q3do.redchisq
            dof[i, j] = q3do.dof
            ndata[i, j] = len(q3do.gd_indx)
            maxncomp[i,j] = q3do.maxncomp

            # Collate continuum data
            if q3do.docontfit:


                # Add wavelength to output cubes
                if firstcontfit:
                    contcube['wave'] = q3do.wave
                    firstcontfit = False

                contcube['npts'][i, j] = len(q3do.fitran_indx)

                # process continuum parameters
                q3do.sepcontpars(q3di)

                hostcube['dat'][i, j, q3do.fitran_indx] = q3do.cont_dat
                hostcube['err'][i, j, q3do.fitran_indx] = err[q3do.fitran_indx]
                hostcube['dq'][i, j, q3do.fitran_indx] = dq[q3do.fitran_indx]
                hostcube['norm_div'][i, j, q3do.fitran_indx] \
                    = np.divide(q3do.cont_dat, q3do.cont_fit)
                hostcube['norm_sub'][i, j, q3do.fitran_indx] \
                    = np.subtract(q3do.cont_dat, q3do.cont_fit)

                if q3di.decompose_ppxf_fit:
                    # Total flux from different components
                    contcube['all_mod'][i, j, q3do.fitran_indx] = q3do.cont_fit
                    contcube['stel_mod'][i, j, q3do.fitran_indx] = \
                        q3do.cont_fit_stel
                    contcube['poly_mod'][i, j, q3do.fitran_indx] = \
                        q3do.cont_fit_poly
                    contcube['stel_sigma'][i, j] = q3do.ct_ppxf_sigma
                    contcube['stel_z'][i, j] = q3do.zstar

                    if q3do.ct_ppxf_sigma_err is not None:
                        contcube['stel_sigma_err'][i, j, :] = \
                            q3do.ct_ppxf_sigma_err
                    if q3do.zstar_err is not None:
                        contcube['stel_z_err'][i, j, :] = q3do.zstar_err

                elif q3di.decompose_qso_fit:
                    if q3di.fcncontfit == 'fitqsohost':
                        if 'refit' in q3di.argscontfit and \
                            'args_questfit' not in q3di.argscontfit:

                            contcube['stel_sigma'][i, j] = \
                                q3do.ct_coeff['ppxf_sigma']
                            contcube['stel_z'][i, j] = q3do.zstar

                            if q3do.ct_ppxf_sigma_err is not None:
                                contcube['stel_sigma_err'][i, j, :] \
                                    = q3do.ct_ppxf_sigma_err
                            if q3do.zstar_err is not None:
                                contcube['stel_z_err'][i, j, :] \
                                    = q3do.zstar_err

                    elif q3di.fcncontfit == 'questfit':

                        contcube['all_mod'][i, j, q3do.fitran_indx] = \
                            q3do.cont_fit
                        contcube['stel_z'][i, j] = q3do.zstar
                        if q3do.zstar_err is not None:
                            contcube['stel_z_err'][i, j, :] = q3do.zstar_err
                else:
                    contcube['all_mod'][i, j, q3do.fitran_indx] = \
                            q3do.cont_fit

                # continuum attenuation
                if q3do.ct_ebv is not None:
                    contcube['stel_ebv'][i, j] = q3do.ct_ebv

                if q3di.decompose_qso_fit:

                    hostcube['dat'][i, j, q3do.fitran_indx] \
                        = q3do.cont_dat - q3do.qsomod
                    contcube['qso_mod'][i, j, q3do.fitran_indx] = \
                        q3do.qsomod
                    # contcube['qso_poly_mod'][i, j, q3do.fitran_indx] = \
                    #     q3do.qsomod_polynorm
                    contcube['host_mod'][i, j, q3do.fitran_indx] = \
                        q3do.hostmod
                    contcube['poly_mod'][i, j, q3do.fitran_indx] = \
                        q3do.polymod_refit

                # if 'remove_scattered' in q3di:
                #     contcube['host_mod'][i, j, q3do.fitran_indx] -= \
                #         q3do.polymod_refit

    # Save emission line and continuum dictionaries
    if q3di.dolinefit:
        outfile = '{0.outdir}{0.label}'.format(q3di)+'.line.npz'
        if sufix is != ''
            sufix1 = '.{sufix}.npz'.replace('..', '.')
            outfile = outfile.replace('.npz', sufix1)
        np.savez(outfile,
                 emlwav=emlwav, emlwaverr=emlwaverr,
                 emlsig=emlsig, emlsigerr=emlsigerr,
                 emlflx=emlflx, emlflxerr=emlflxerr,
                 emlweq=emlweq, emlncomp=emlncomp,
                 ncols=cube.ncols, nrows=cube.nrows,
                 ndata=ndata, dof=dof, redchisq=redchisq, maxncomp=maxncomp)
        print('q3dcollect: Saving emission-line fit results into '+outfile)
    if q3di.docontfit:
        outfile = '{0.outdir}{0.label}'.format(q3di)+'.cont.npy'
        np.save(outfile, contcube)
        print('q3dcollect: Saving continuum fit results into '+outfile)
