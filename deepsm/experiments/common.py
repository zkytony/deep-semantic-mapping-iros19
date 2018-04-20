# Shared parameters

import numpy as np

# 56x21
# resolution      = 0.02
# num_angle_cells = 56
# min_radius      = 0.3
# max_radius      = 5
# radius_factor   = 1.15

# 32x8
resolution      = 0.02
num_angle_cells = 32
min_radius      = 0.3
max_radius      = 5
radius_factor   = 1.874

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
GROUNDTRUTH_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/dataset/cold-groundtruth"
COLD_ROOT = "/home/zkytony/sara/sara_ws/src/sara_processing/sara_cold_processing/forpub/COLD"
GRAPHSPN_RESULTS_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/results/graphspn"
DGSM_RESULTS_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/results/dgsm/4classes_results"
TOPO_MAP_DB_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/dataset/topo_map"
DGSM_DB_ROOT = "/home/zkytony/Documents/thesis/experiments/deep-semantic-mapping/deepsm/experiments/dataset/dgsm"

# BP_EXEC_PATH = "/home/zkytony/Documents/thesis/experiments/spn_topo/spn_topo/tests/tbm/factor_graph/ft"
# BP_RESULT_DIR = "/home/zkytony/Documents/thesis/experiments/spn_topo/spn_topo/tests/tbm/factor_graph/r"

# Laptop
# COLD_ROOT = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/data/groundtruth"
# RESULTS_DIR = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/results/"
# TOPO_MAP_DB_ROOT = "/home/kaiyu/Documents/Projects/Research/SaraRobotics/repo/spn_topo/experiments/data/topo_map"
