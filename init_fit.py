import os
import numpy as np
from q3dfit.q3din import q3din
from q3dfit.q3dutil import get_q3dio

def main():
    # set up directory structure
    indir = "input/"
    outdir = "output/"
    label = "4C1971"

    if not os.path.exists(indir):
        os.makedirs(indir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logfile = os.path.join(indir, label+"-fitlog.txt")
    infile = "4C1971_with_data_quality.fits"

    # 
    q3di = q3din(infile, label, outdir=outdir, logfile=logfile)
    q3di.argsreadcube = {"wmapext": True}
    cube = q3di.load_cube()
    q3di.name = label
    q3di.zsys_gas = 3.5892
    q3di.fitrange = [2.2, 2.35]

    q3di = init_fit(q3di)

    np.save(os.path.join(indir, "q3di.npy"), q3di)
    # print(q3di.__dict__)

def init_fit(q3dii: "q3din | str") -> q3din:
    q3di: q3din = get_q3dio(q3dii)

    #
    lines = ['Hbeta', '[OIII]4959', '[OIII]5007']
    q3di.init_linefit(lines, linetie='[OIII]5007', maxncomp=1, checkcomp=False)
    q3di.siglim_gas = np.array([40., 2000.])
    q3di.spect_convol['ws_instrum'] = {'JWST_NIRSPEC':['G235H']}
    q3di.argslinefit['method'] = 'least_squares'
    q3di.argslinefit['ftol'] = 1.e-8
    q3di.argslinefit['gtol'] = 1.e-8
    q3di.argslinefit['xtol'] = 1.e-8
    q3di.argslinefit['x_scale'] = 'jac'
    q3di.argslinefit['tr_solver'] = 'lsmr'
    q3di.argscheckcomp['sigcut'] = 3.
    q3di.argscheckcomp['ignore']= ['Hbeta']

    # 
    q3di.init_contfit("fitpoly")
    q3di.argscontfit.update({"fitord": 1})
    q3di.maskwidths_def = 2000
    q3di.masksig_secondfit = 2

    return q3di

if __name__ == "__main__":
    main()