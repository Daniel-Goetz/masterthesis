# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# astropy
from astropy import units as u

# q3dfit
from q3dfit import q3dutil

# bruno
from bq3dfit import bq3dplots, bq3dutils, bCOSMICube
from importlib import reload


#=======================================
outdir_maps = f'maps_new/'
filename_q3di = 'q3di_4c0324_oiii-hbeta_6gauss_merge_master-aic0_SNRcut3_AICcut10.npy'
filenme_npz = None
#filename_npz = '4c0324_oiii-hbeta_6gauss_merge_master-aic0_SNRcut3_AICcut10.line.npz'

q3di = q3dutil.get_q3dio(filename_q3di)
cube = q3di.load_cube()






# ==========================
# Define parameters for function
opts_load_maps = dict(
                    SNR_cut = 3,
                    which_snr_selection = 'pkflux',
                    which_pkflux_noise = 'fitsfile',
                    fluxpk_noise_fitsfile = None,
                    #which_flux_noise = 'lmfit',
                    apply_mask = True,
                    stats = True,
                    sigmin = 10,
                    transpose = true,
                    redchisqmax = 200,
                    type_mask2D = 'any',
                    velmin = None,
                    velmax = None,
                    cube=cube,
                    )

opts_load_maps['redshift'] = 3.5657



#plot_opts_vel = {
#                 'vmin': 1000, 'vmax': 1200,
#                 'vpercent':None,
#                 }

plot_opts_vel = {}

bCOSMICube.plot_maps_line(filename_q3di, cube=cube, 
                        filename_npz=filename_npz,
                        output_dir=outdir_maps,
                        opts_load_maps=opts_load_maps,
                        n_gauss=n_cluster_final,
                        plot_opts_vel=plot_opts_vel)

