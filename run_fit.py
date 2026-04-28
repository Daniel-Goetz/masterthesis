q3di = "input/q3di.npy"
cols = None#[1,10]#[15,30]
rows = None#[1,10]#[30,45]
# from q3dfit.q3dutil import get_Cube, get_q3dio
# cube, _ = get_Cube(get_q3dio(q3di))
# cols = [1, cube.ncols]
# rows = [1, cube.nrows]
# print(cols, rows)

# # fit
# from q3dfit.q3df import q3dfit
# q3dfit(q3di, cols=cols, rows=rows, ncores=22, quiet=False, nocrash=True)

# Collect
from q3dfit.q3dcollect import q3dcollect
q3dcollect(q3di, cols=cols, rows=rows, compsortpar="wave", quiet=True)