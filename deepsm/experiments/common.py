# Shared parameters

import numpy as np

resolution      = 0.02
num_angle_cells = 56
min_radius      = 0.3
max_radius      = 5
radius_factor   = 1.15

angles = np.linspace(-180, 180, num_angle_cells+1)
r=min_radius
radiuses=[r]
v = 0.04
while r<max_radius:
    r = r+v
    radiuses.append(r)
    v*=radius_factor
radiuses = np.array(radiuses)
num_radius_cells = len(radiuses)-1

