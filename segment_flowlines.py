import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


from flow_utils import *




def surface_to_feature_vector(surfaces, max_size=None, only_y=False):
    # create a feature vector from the x,y coordinates of the surfaces

    if max_size == None:
        max_size = 0
    for surface in surfaces:
        max_size = max(max_size, surface.path.shape[0])

    feature_vectors = []
    for surface in surfaces:
        template = np.zeros((max_size, 2)) - 1
        template[:surface.path.shape[0], :] = surface.path
        feature_vectors.append(template)

    feature_vectors = np.array(feature_vectors)

    if only_y:
        feature_vectors = feature_vectors[..., 0]

    feature_vectors = feature_vectors.reshape(len(feature_vectors), -1)

    return feature_vectors, max_size

def sort_tops(seismic_slice, surfaces, clusterer):

    x = np.arange(seismic_slice.shape[1])
    y = np.ones(seismic_slice.shape[1]) * (seismic_slice.shape[0] - 1)
    boundary_base = np.stack((y,x)).T
    y = np.zeros(seismic_slice.shape[1])
    boundary_top = np.stack((y,x)).T

    label_boundaries = {}
    unique_labels = np.unique(clusterer.labels_)
    for label in unique_labels:
        label_boundaries[label] = [np.inf, -np.inf, 0, 0]


    for surface in surfaces:
        mean_depth = surface.path[:,0].mean()
        if mean_depth < label_boundaries[surface.kmeans_label][0]:
            label_boundaries[surface.kmeans_label][2] = surface.path
            label_boundaries[surface.kmeans_label][0] = mean_depth

        if mean_depth > label_boundaries[surface.kmeans_label][1]:
            label_boundaries[surface.kmeans_label][1] = mean_depth
            label_boundaries[surface.kmeans_label][3] = surface.path



    mean_depth_tops = []
    tops = []


    for key,values in label_boundaries.items():
        mean_depth_top, mean_depth_base, top, base = values
        mean_depth_tops.append(mean_depth_top)
        tops.append(top)

    tops.append(boundary_top)
    mean_depth_tops.append(0)

    tops.append(boundary_base)
    mean_depth_tops.append(seismic_slice.shape[0])



    sort_inds = np.argsort(mean_depth_tops)
    sorted_tops = [tops[ind] for ind in sort_inds]
    return sorted_tops

from scipy.interpolate import interp1d

def interpolate_boundary(boundary_path, image_shape):
    """
    Interpolate a boundary so it spans all x coordinates in the image.
    If the interpolated y value is out of bounds, snap it to the nearest valid value.

    :param boundary_path: Array of boundary points [y, x].
    :param image_shape: Tuple of (height, width) of the image.
    :return: Interpolated boundary path.
    """
    height, width = image_shape

    # Extract x and y coordinates
    x_coords = boundary_path[1:-1, 1]
    y_coords = boundary_path[1:-1, 0]

    # Create an interpolation function
    interp_func = interp1d(x_coords, y_coords, kind='linear', fill_value="extrapolate")

    # New x coordinates spanning the entire width
    new_x_coords = np.arange(width)

    # Interpolated y coordinates
    new_y_coords = interp_func(new_x_coords)

    new_y_coords[new_y_coords < 0] = 0
    new_y_coords[new_y_coords > height] = height

    # Combine x and y coordinates
    interpolated_path = np.vstack((new_y_coords, new_x_coords)).T

    return interpolated_path

def segment_flowlines(seismic_line, sample_interval_x, mode, num_clusters=None, only_y=False, cmap=plt.cm.jet):

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
    array_mask = np.zeros(seismic_line.shape)
    if num_clusters:
        scaler = StandardScaler()
        X, _ = surface_to_feature_vector(surfaces_seismic, only_y=only_y)
        clusterer = KMeans(n_clusters=num_clusters, random_state=0,
                           n_init="auto").fit(scaler.fit_transform(X))

        #scalar_map = create_scalar_colormap(np.arange(num_clusters),cmap=cmap)
        for ind, surface in enumerate(surfaces_seismic):
            surface.kmeans_label = clusterer.labels_[ind]

        #sorted_labels
        centroids = clusterer.cluster_centers_

        # Step 2: Sort centroids based on the y-coordinate (or your chosen criterion)
        sorted_indices = np.argsort(centroids[:, 1])  # Assuming y-coordinate is at index 1
        sorted_indices = np.append(sorted_indices, num_clusters + 1)
        # Step 3: Create mapping from old to new labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}

        # Step 4: Reassign labels to surfaces
        for surface in surfaces_seismic:
            surface.kmeans_label = label_mapping[surface.kmeans_label]
        
        label_map_back = {value:key for key,value in label_mapping.items()}

        sorted_tops = sort_tops(seismic_line, surfaces_seismic, clusterer)
        sorted_tops_surfaces = [Surface(tops) for tops in sorted_tops]
        interp_tops = [Surface(interpolate_boundary(top.path, seismic_line.shape)) for top in sorted_tops_surfaces]

        _, Y = np.meshgrid(np.arange(seismic_line.shape[1]), np.arange(seismic_line.shape[0]))
        

        for i in range(len(interp_tops) - 1):
            
            y1=interp_tops[i].path[:, 0]
            y2=interp_tops[i+1].path[:, 0]


            array_mask[np.where((Y > y1)&(Y < y2))] = label_map_back[i]

    return array_mask
        
   

# Path to 2D seismic
seispath = 'F3_2Dline.npy'
seismic_line = np.load(seispath)

# the distance between traces to sample peaks/troughs from
sample_interval_x = 100  

# if we are picking on both peak and trough in each trace
mode = 'both'  


#only_y : if we use both the x and y coordinate of the surface paths. Defaults to both. 
only_y = False

#! Clustering parameters
# number of clusters to use for clustering.
num_clusters = 10  


segmentation_map = segment_flowlines(seismic_line,
                                sample_interval_x=sample_interval_x, mode=mode, only_y=only_y, num_clusters=num_clusters)

"""
fig,ax = plt.subplots(1,1,dpi=250)
ax.imshow(segmentation_map,cmap='jet')
plt.show()
"""