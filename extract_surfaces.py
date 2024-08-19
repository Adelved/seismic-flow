import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d
import matplotlib.pyplot as plt

import shapely
from flow_utils import *




# Path to 2D seismic
#Assumes that the seismic is in the format [samples, traces]
seispath = 'F3_2Dline.npy'
# the distance between traces to sample peaks/troughs from
sample_interval_x = 100  
# if we are picking on both peak and trough in each trace
mode = 'both'  
sigma = 1 #smoothning
rho = 1 #area to calculate gradients

seismic_line = np.load(seispath)
S = structure_tensor_2d(seismic_line, sigma=sigma, rho=rho)
_, vector_array = eig_special_2d(S)

peak_params = {
    "height": None,
    "distance": None,
    "prominence": None
}

surfaces, _ = extract_surfaces(
    seismic_line,

    vector_array,

    [sample_interval_x],

    mode=mode,

    kwargs=peak_params
)


fig,ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(seismic_line,cmap='gray', extent=[0, seismic_line.shape[1], seismic_line.shape[0], 0], alpha=.0)

#surface objects have a 2D numpy array attribute path (y, x)
#adjust alpha (0-1) to get overlap effect
for surface in surfaces:
    color = 'k'
    ax.plot(surface.path[:,1], surface.path[:,0], linewidth=1, color = color , alpha = .5)
plt.show()
