import numpy as np
from q3dfit.q3dpro import OneLineData, LineData
from q3dfit import q3dutil
import bq3dplots

# q3di saved configuration file
filename_q3di = "input/q3di.npy"

# Open
q3di = q3dutil.get_q3dio(filename_q3di)

# -- Generating a map that will be used as a base image to locate spaxels --
# linedata
linedat = LineData(q3di)
# Read [OIII]
o3data = OneLineData(linedat, '[OIII]5007')
# Total [OIII] flux
# It can be any image with the same cube spatial size, but notice that I
# transposed the map generated from q3dfit 
Fo3_tot = np.nansum(o3data.flux.T, axis=0)

# Interactive plot!
bq3dplots.interactive_plot(filename_q3di, img=Fo3_tot)

