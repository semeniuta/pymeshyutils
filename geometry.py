import numpy as np


def triangle_sides_distances(points):

    p1, p2, p3 = points

    d1 = np.linalg.norm(p1 - p2)
    d2 = np.linalg.norm(p2 - p3)
    d3 = np.linalg.norm(p3 - p1)

    return [d1, d2, d3]


def triangle_area(points):

    d1, d2, d3 = triangle_sides_distances(points)

    s = 0.5 * (d1 + d2 + d3)

    return np.sqrt(s * (s - d1) * (s - d2) * (s - d3))


def normalize_to_unit(v):
    return v / np.linalg.norm(v)


def e2h(x):
    return np.array([el for el in x] + [1.0])


def h2e(x):
    return x[:-1] / x[-1]


def normal_vector(points):

    a, b, c = points

    ab = b - a
    ac = c - a

    n = np.cross(ab, ac)

    return normalize_to_unit(n)


def angle_cos_between_two_vectors(v1, v2):
    
    norm_const = np.linalg.norm(v1) * np.linalg.norm(v2)

    return np.dot(v1, v2) / norm_const


def triangle_angles_cos(points):

    def other_indices(i):
        
        if i == 0:
            return 1, 2

        if i == 1:
            return 0, 2

        if i == 2:
            return 0, 1

    def vectors_from_pt(i):

        other = other_indices(i)
        return [points[j] - points[i] for j in other]

    angles = []
    for i in range(3):

        v1, v2 = vectors_from_pt(i)
        theta = angle_cos_between_two_vectors(v1, v2)
        angles.append(theta)

    return angles




        
    





    
