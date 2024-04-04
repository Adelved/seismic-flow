import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d
import matplotlib.pyplot as plt

import shapely
from flow_utils import *



def paths_intersecting(path1, path2, threshold):
    # Convert paths to sets for efficient intersection operation
    set_path1 = set(path1)
    set_path2 = set(path2)

    # Calculate intersection
    intersection = set_path1 & set_path2

    # Calculate the percentage of overlap
    overlap_percentage = (len(intersection) / len(set_path1)) * 100

    # Check if the overlap percentage is greater than or equal to the threshold
    return overlap_percentage >= threshold

#calculate the percentage overlap between two linestring objects
def degree_of_intersection(existing, test):
  intersection = existing.intersection(test)

  points = shapely.count_coordinates(intersection)

  return  points / shapely.count_coordinates(test)

def prune_overlapping_surfaces(surfaces_seismic_sorted, linewidth=1, overlap_threshold=0.95):
    non_overlapping = []
    non_overlapping.append(surfaces_seismic_sorted[0])


    for surface in surfaces_seismic_sorted:
        intersects = False
        for existing in non_overlapping:
            existing_line = existing.linestring.buffer(linewidth)
            testline = surface.linestring
            perc_intersect = degree_of_intersection(existing_line, testline)

            if perc_intersect > overlap_threshold:
                intersects = True
                break
        if intersects == False:
            non_overlapping.append(surface)
    
    return non_overlapping




def create_heatmap(surfaces, seismic_slice):
    heatmap = np.zeros(seismic_slice.shape)

    for surface in surfaces:
        y = np.round(surface.path[:, 0], 0)
        x = np.round(surface.path[:, 1], 0)

        invalid_y_inds = np.where(y == seismic_slice.shape[0])[0]
        invalid_x_inds = np.where(x == seismic_slice.shape[1])[0]

        if len(invalid_y_inds) > 0:
            y[invalid_y_inds] = seismic_slice.shape[0] - 1

        if len(invalid_x_inds) > 0:
            x[invalid_x_inds] = seismic_slice.shape[1] - 1

        heatmap[y.astype(int), x.astype(int)] += 1

    return heatmap


def generate_surfaces_2D(seismic_line, sample_interval_x, overlap_threshold = 0.95, linewidth =1,  mode = 'both'):

    S = structure_tensor_2d(seismic_line, sigma=1.0, rho=1.0)
    _, vector_array = eig_special_2d(S)

    peak_params = {
        "height": None,
        "distance": None,
        "prominence": None
    }

    surfaces_seismic, _ = extract_surfaces(
        seismic_line,

        vector_array,

        [sample_interval_x],

        mode=mode,

        kwargs=peak_params
    )

    # calculate the heatmap
    heatmap = create_heatmap(surfaces_seismic, seismic_line)

    # calculate surface mean heat
    for surface in surfaces_seismic:
        surface.mean_heat = np.mean(
            heatmap[surface.path[:, 0].astype(int), surface.path[:, 1].astype(int)])

    # sort on mean heat
    surfaces_seismic_sorted = list(
        reversed(sorted(surfaces_seismic, key=lambda obj: obj.mean_heat)))

    return prune_overlapping_surfaces(surfaces_seismic_sorted, linewidth, overlap_threshold)

def surface2array(seismic_line, surfaces, num_surfaces=None):
    surface_array = np.zeros(seismic_line.shape)

    if num_surfaces == None:
        num_surfaces = len(surfaces)

    for surface in surfaces[:num_surfaces]:
        
        surface_array[surface.path[:, 0].astype(
            int), surface.path[:, 1].astype(int)] = 1

    return surface_array

#! Function expects input as a numpy array with the time on the first axis=0

# Path to 2D seismic
seispath = 'F3_2Dline.npy'
seismic_line = np.load(seispath)

# the distance between traces to sample peaks/troughs from
sample_interval_x = 100  

# if we are picking on both peak and trough in each trace
mode = 'both'  

# The thickness to expand the lines when calculating the overlap. Higher values result in more aggressive pruning of flow lines (5-10) are good start values.
linewidth = 7

# overlap tolerance for surfaces. Removes surfaces that have a higher overlap than 0.95
overlap_threshold = 0.95

surfaces = generate_surfaces_2D(seismic_line, overlap_threshold=overlap_threshold,
                                sample_interval_x=sample_interval_x, mode=mode, linewidth=linewidth)

surface_array = surface2array(seismic_line, surfaces)

"""
fig,ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(seismic_line,cmap='gray', extent=[0, seismic_line.shape[1], seismic_line.shape[0], 0], alpha=.0)

for surface in surfaces:
    color = 'k'
    ax.plot(surface.path[:,1], surface.path[:,0], linewidth=1, color = color , alpha = 1)
plt.show()


fig,ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(surface_array,cmap='gray', extent=[0, seismic_line.shape[1], seismic_line.shape[0], 0])
plt.show()
"""