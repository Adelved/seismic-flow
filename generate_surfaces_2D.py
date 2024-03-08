import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d
import scipy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# surface class


class Surface():
    def __init__(self, path, x_seed=None):
        self.x_seed = x_seed
        self.path = path
        self.label = None
        self.tuple_path = [tuple(

            (int(np.round(self.path[i, 0], 0)),
             int(np.round(self.path[i, 1], 0)))
        )
            for i in range(len(self.path))]

        self.line_weight = np.ones(len(self.path))

    def create_weighted_path(self):
        x = self.path[:, 1]
        y = self.path[:, 0]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments


def vector_field(x, y, vector_array):
    y = int(y)
    x = int(x)
    y = min(y, vector_array.shape[1] - 1)
    x = min(x, vector_array.shape[2] - 1)
    v, u = vector_array[:, y, x]

    return u, v


def runge_kutta_4(x0, y0, h, steps, vector_array, num_decimals=2):
    # Fourth-order Runge-Kutta method
    path_x = [x0]
    path_y = [y0]

    for _ in range(steps):
        u0, v0 = vector_field(x0, y0, vector_array)
        k1_x = h * u0
        k1_y = h * v0

        u1, v1 = vector_field(x0 + 0.5 * k1_x, y0 + 0.5 * k1_y, vector_array)
        k2_x = h * u1
        k2_y = h * v1

        u2, v2 = vector_field(x0 + 0.5 * k2_x, y0 + 0.5 * k2_y, vector_array)
        k3_x = h * u2
        k3_y = h * v2

        u3, v3 = vector_field(x0 + k3_x, y0 + k3_y, vector_array)
        k4_x = h * u3
        k4_y = h * v3

        x0 += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        y0 += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6

        path_x.append(x0)
        path_y.append(y0)

    path_x = np.array(path_x)
    path_y = np.array(path_y)

    path_x = np.round(path_x, num_decimals)
    path_y = np.round(path_y, num_decimals)
    return path_x, path_y


def extract_surfaces(seismic_slice, vector_array, sample_intervals, mode='peak', kwargs={}):
    for sample_interval_x in sample_intervals:

        num_peaks, num_troughs = 0, 0

        seeds = np.arange(0, seismic_slice.shape[1] - 1, sample_interval_x)
        num_decimals = 1

        surfaces = []
        for x_seed in seeds:

            if mode == 'peak':
                trace = seismic_slice[:, x_seed].copy()
                peaks, _ = scipy.signal.find_peaks(trace, **kwargs)

            if mode == 'trough':
                trace = -seismic_slice[:, x_seed].copy()
                peaks, _ = scipy.signal.find_peaks(trace, **kwargs)

            if mode == 'both':
                trace = seismic_slice[:, x_seed].copy()
                peaks, _ = scipy.signal.find_peaks(np.abs(trace), **kwargs)

            if len(peaks) == 0:
                continue

            h = 1

            for ind, peak in enumerate(peaks):

                # Initial position
                y0, x0 = peak, x_seed

                # Number of steps
                steps = vector_array.shape[2] - x_seed

                # Get the streamline for flipped array
                path_x_flip, path_y_flip = runge_kutta_4(
                    x0, y0, h, x_seed, vector_array * -1, num_decimals=num_decimals)

                path_y_flip = path_y_flip[(path_x_flip > 0) & (
                    path_x_flip < seismic_slice.shape[1])]
                path_x_flip = path_x_flip[(path_x_flip > 0) & (
                    path_x_flip < seismic_slice.shape[1])]

                path_x_flip = path_x_flip[(path_y_flip > 0) & (
                    path_y_flip < seismic_slice.shape[0])]
                path_y_flip = path_y_flip[(path_y_flip > 0) & (
                    path_y_flip < seismic_slice.shape[0])]

                # Get the streamline
                path_x, path_y = runge_kutta_4(
                    x0, y0, h, seismic_slice.shape[1] - x_seed, vector_array, num_decimals=num_decimals)

                path_y = path_y[(path_x > 0) & (
                    path_x < seismic_slice.shape[1])]
                path_x = path_x[(path_x > 0) & (
                    path_x < seismic_slice.shape[1])]

                path_x = path_x[(path_y > 0) & (
                    path_y < seismic_slice.shape[0])]
                path_y = path_y[(path_y > 0) & (
                    path_y < seismic_slice.shape[0])]

                path_flip = list(zip(path_y_flip, path_x_flip))
                path = list(zip(path_y, path_x))

                # Both lists are non-empty
                if len(path) > 0 and len(path_flip) > 0:
                    merged_path = np.concatenate(
                        (list(reversed(path_flip)), path))

                # Only path is non-empty
                elif len(path) > 0:
                    merged_path = np.array(path)

                # Only path_flip is non-empty
                elif len(path_flip) > 0:
                    merged_path = np.array(path_flip)

                # Both lists are empty
                else:
                    merged_path = []  # or handle as needed

                surface = Surface(merged_path, x_seed)

                surfaces.append(surface)
    return surfaces, num_peaks + num_troughs


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


def generate_surfaces_2D(seismic_line, sample_interval_x, overlapp_threshold, mode, num_clusters=None, only_y=False):

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

    if num_clusters:
        scaler = StandardScaler()
        X, _ = surface_to_feature_vector(surfaces_seismic, only_y=only_y)
        clusterer = KMeans(n_clusters=num_clusters, random_state=0,
                           n_init="auto").fit(scaler.fit_transform(X))

        for ind, surface in enumerate(surfaces_seismic):
            surface.label = clusterer.labels_[ind]

    # calculate the heatmap
    heatmap = create_heatmap(surfaces_seismic, seismic_line)

    # calculate surface mean heat
    for surface in surfaces_seismic:
        surface.mean_heat = np.mean(
            heatmap[surface.path[:, 0].astype(int), surface.path[:, 1].astype(int)])

    # sort on mean heat
    surfaces_seismic_sorted = list(
        reversed(sorted(surfaces_seismic, key=lambda obj: obj.mean_heat)))

    overlapping_paths = []
    overlapping_paths.append(surfaces_seismic_sorted[0])

    for surface in surfaces_seismic_sorted:
        intersects = False
        for existing in overlapping_paths:
            if paths_intersecting(surface.tuple_path, existing.tuple_path, threshold=overlapp_threshold):
                intersects = True
                break

        if not intersects:
            overlapping_paths.append(surface)
    return overlapping_paths

def surface2array(seismic_line, surfaces, num_surfaces):
    surface_array = np.zeros(seismic_line.shape)

    for i in range(num_surfaces):
        surface = surfaces[i]
        if surface.label:
            label = surface.label
        else:
            label = 1

        surface_array[surface.path[:, 0].astype(
            int), surface.path[:, 1].astype(int)] = label

    return surface_array


#! Function expects input as a numpy array with the time on the first axis=0

# Path to 2D seismic
seispath = 'F3_2Dline.npy'
seismic_line = np.load(seispath)

# the distance between traces to sample peaks/troughs from
sample_interval_x = 100  

# if we are picking on both peak and trough in each trace
mode = 'both'  

# overlap tolerance for surfaces. Removes surfaces that have a higher overlap than 25 %
overlapp_threshold = 25

# number of top surfaces to plot
num_surfaces = 10  





#! Clustering parameters
# number of clusters to use for clustering.
num_clusters = 10  

surfaces = generate_surfaces_2D(seismic_line, overlapp_threshold=overlapp_threshold,
                                sample_interval_x=sample_interval_x, mode=mode, num_clusters=num_clusters)

surface_array = surface2array(seismic_line, surfaces, num_surfaces)



"""
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(surface_array, cmap='jet')
plt.show()
"""