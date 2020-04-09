import numpy as np
from matplotlib import pyplot as plt
import pymeshy


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
        
        
