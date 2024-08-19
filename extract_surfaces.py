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

