import numpy as np

def generate_weights(c, r, resolution = 800, n_rings = 20):
    cx, cy = c.squeeze()
    circle = []
    radius = []
    curr_radius = 0
    while curr_radius <= r:
        curr_reso = int(resolution * curr_radius / r)
        for i in range(curr_reso if curr_reso else 1):
            circle.append((curr_radius * np.cos(i) + cx, curr_radius * np.sin(i) + cy))
            radius.append(curr_radius)
        curr_radius += r / n_rings
    return np.array(circle)[:,np.newaxis], np.array(radius)