from q3dfit.q3dpro import OneLineData, ContData, LineData
from q3dfit import q3dutil
from bq3dfit import bq3dplots1

def load_linemaps(q3di, line = '[OIII]5007'):

    from astropy import constants
    c = constants.c.to('km/s')

    linedat = LineData(q3di)
    linelist = q3dutil.get_linelist(q3di)
    rest_line = linelist[linelist['name'] == line]['lines'][0]

    linedata = OneLineData(linedat, line)
    flux = linedata.flux.T
    amp = linedata.pkflux.T
    vel = (linedata.wave.T / (1 + q3di.zsys_gas) / rest_line - 1) * c
    sig = linedata.sig.T

    #mask_dq = cube.dq.astype(bool).T.all(axis=0)
    #mask_sig = (sig == 0).all(axis=0)
    #mask_amp = (amp == 0).all(axis=0)
    mask_dq = cube.dq.astype(bool).T.sum(axis=0)
    mask_sig = sig == 0
    mask_amp = amp == 0
    mask = mask_sig + mask_amp
    mask2D = mask.all(axis=0)

    flux[mask] = np.nan
    sig[mask] = np.nan
    vel[mask] = np.nan

    linedat.flux = flux
    linedat.vel = vel
    linedat.sig = sig

    return linedat

