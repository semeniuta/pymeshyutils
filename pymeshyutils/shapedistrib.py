import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pymeshy
from .geometry import triangle_sides_distances, triangle_angles_cos, triangle_area


def hist(x, n_bins, min_val, max_val, normalize=False):

    counts, bin_edges = np.histogram(x, bins=n_bins, range=(min_val, max_val))

    if normalize:
        counts = counts / len(x)

    bin_centers = compute_bin_centers(bin_edges)

    return counts, bin_edges, bin_centers


def compute_bin_centers(bin_edges):
    n = len(bin_edges) - 1
    return [0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(n)]


def hist_abs_diff(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))


def hist_diff_std(hist1, hist2):
    return (hist1 - hist2).std()


def several_scalings(facets, scaling_factors, direction=0, n_samples=100000):

    def gen(facets):
        return np.array(pymeshy.generate_d2_samples_for_facets(facets, n_samples))

    all_samples = [
        gen(facets),
    ]

    for s in scaling_factors:

        T = np.eye(4)
        T[direction, direction] = s

        facets_t = pymeshy.transform_facets(facets, T)
        samples_t = gen(facets_t)

        all_samples.append(samples_t)

    return all_samples


def create_histograms(all_samples, **hist_kwargs):

    hist_results = [hist(samples, **hist_kwargs) for samples in all_samples]

    counts = np.array([tpl[0] for tpl in hist_results])
    bin_edges = np.array([tpl[1] for tpl in hist_results])
    bin_centers = np.array([tpl[2] for tpl in hist_results])

    return counts, bin_edges, bin_centers


def cumulative_areas_hist_differences(counts_matrix):

    counts0 = counts_matrix[0, :]

    n = counts_matrix.shape[0]

    cumulative_areas = []
    for i in range(1, n):
        counts = counts_matrix[i, :]
        s = np.sum(np.abs(counts0 - counts))
        cumulative_areas.append(s)

    return cumulative_areas
        

def measurements_from_3points(points_all):

    measurements = np.zeros((len(points_all), 7), dtype=np.float64)

    for i, points in enumerate(points_all):

        d_unsorted = triangle_sides_distances(points)

        indices_order = sorted(range(3), key=lambda i: d_unsorted[i])

        points_new_order = [points[idx] for idx in indices_order]
        d1, d2, d3 = [d_unsorted[idx] for idx in indices_order]

        theta_1, theta_2, theta_3 = triangle_angles_cos(points_new_order)
        area = triangle_area(points)

        measurements[i, :] = area, theta_1, theta_2, theta_3, d1, d2, d2

    return pd.DataFrame(measurements, columns=['area', 'theta_1', 'theta_2', 'theta_3', 'd1', 'd2', 'd3'])
        

def sample_measurements_from_3points(mesh, n_samples, random_state=-1):

    points = np.array(pymeshy.generate_random_points_for_facets(mesh, n_samples, 3, random_state=random_state))
    meas_df = measurements_from_3points(points)

    return meas_df
