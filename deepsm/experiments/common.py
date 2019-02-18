# Shared parameters

import numpy as np
import os

# 56x21
resolution      = 0.02
num_angle_cells = 56
min_radius      = 0.3
max_radius      = 5
radius_factor   = 1.15

# # 32x8
# resolution      = 0.02
# num_angle_cells = 32
# min_radius      = 0.3
# max_radius      = 5
# radius_factor   = 1.874

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


# ----- Constants -----
# Gerlach
COLD_ROOT = "/home/zkytony/sara/sara_ws/src/sara_processing/sara_cold_processing/forpub/COLD"
EXPERIMENTS_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments"

GROUNDTRUTH_ROOT = os.path.join(EXPERIMENTS_ROOT, "dataset", "cold-groundtruth")
TOPO_MAP_DB_ROOT = os.path.join(EXPERIMENTS_ROOT, "dataset", "topo_map")
DGSM_DB_ROOT = os.path.join(EXPERIMENTS_ROOT, "dataset", "dgsm")

BP_RESULTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "results", "factor_graph")
GRAPHSPN_RESULTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "results", "graphspn")
DGSM_RESULTS_ROOT = os.path.join(EXPERIMENTS_ROOT, "results", "dgsm")

BP_EXEC_PATH = os.path.join(EXPERIMENTS_ROOT, "factor_graph", "fg_topo")

# Laptop
# COLD_ROOT = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/data/groundtruth"
# RESULTS_DIR = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/results/"
# TOPO_MAP_DB_ROOT = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/data/topo_map"
