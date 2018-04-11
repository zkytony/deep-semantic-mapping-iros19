import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines

##########PlaceGrid related##########
def computeRadii(radius, num_radial_steps, radial_increment_factor=5.0):
    """
    Return a numpy array of floats, where each float is the length of the corresponding radial
    step from the center of the polar coordinates to the boundary.

    Parameters:
    - radius: the radius of the polar region
    - num_radial_steps: number of steps to take from polar center to boundary.
    - radial_incremental_factor: a real number that controls how dramatic the changes of
                                 radius is as we go from the center of the polar outwards.
    """
    
    r0 = radius / num_radial_steps / radial_increment_factor
    radial_inc = -2*(num_radial_steps*r0 + r0 - radius)/(num_radial_steps*(num_radial_steps+1))
    # radii stores an array of polar cell length, in meter.
    radii = np.zeros(num_radial_steps)
    for i in range(num_radial_steps):
        radii[i] = r0  + i*radial_inc
    return radii


def pick_id(numbers, seed):
    """
    Pick an integer based on seed that is unique in numbers, a set of integers.
    """
    while seed in numbers:
        seed += 1
    return seed

    
def transform_coordinates(gx, gy, map_spec, img):
    # Given point (gx, gy) in the gmapping coordinate system (in meters), convert it
    # to a point or pixel in Cairo context. Cairo coordinates origin is at top-left, while
    # gmapping has coordinates origin at lower-left.
    imgHeight, imgWidth = img.shape
    res = float(map_spec['resolution'])
    originX = float(map_spec['origin'][0])  # gmapping map origin
    originY = float(map_spec['origin'][1])
    # Transform from gmapping coordinates to pixel cooridnates.
    return ((gx - originX) / res, imgHeight - (gy - originY) / res)




class ColdDatabaseManager:
    # Used to obtain cold database file paths

    def __init__(self, db_name, db_root):
        self.db_root = db_root
        self.db_name = db_name

    def groundtruth_file(self, floor, filename):
        return os.path.join(self.db_root, self.db_name, 'groundtruth', floor, filename)
