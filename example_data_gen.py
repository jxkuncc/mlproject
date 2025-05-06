"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: April 28, 2025
"""

from time import time
import numpy as np
from copy import deepcopy
from shapelib.Shapes import Shape
from shapelib.Data import data_from_shape

# Don't accidentally run this script
exit(0)

out_dir = 'data/10cube/'
out_prefix = '10cube_'

train_samples = 5000
test_samples =  5000
rng = np.random.default_rng(1236)

# Create/ Read a Nominal Shape
cube10 = Shape.from_file('data/10cube/10cube.json')


# Variable to see how long this takes
start_time = time()

#Generate Training Data
data_from_shape(
    nominal_shape = cube10, 
    num_samples = train_samples, 
    out_dir = out_dir+'/train',
    out_prefix= out_prefix+'train',
    rng=rng)

#Generate Test Data
data_from_shape(
    nominal_shape = cube10, 
    num_samples = test_samples, 
    out_dir = out_dir+'/test',
    out_prefix= out_prefix+'test',
    rng=rng)

# Finally, tells how long it took
print(f'Generated {train_samples} training samples and {test_samples} test samples of {repr(cube10).lower()} in {time() -start_time} seconds.')




