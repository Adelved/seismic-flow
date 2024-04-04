import numpy as np
import scipy
from shapely.geometry import LineString

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
        
        self.linestring = LineString([(y,x) for y,x in list(zip(self.path[:,0],self.path[:,1]))])
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