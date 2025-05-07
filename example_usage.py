"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: May 7, 2025
"""
import numpy as np
from shapelib.Shapes import Shape, polycube
from shapelib.Fits import weighted_fit, best_fit_transform, apply_htm
from shapelib.Models import tobatch, frombatch
import torch

# Load the nominal shape
cube10 = Shape.from_file('data/10cube/10cube.json')
print('The nominal shape is', repr(cube10).lower())
print('The nominal vertices [x,y,z]\n', cube10.vertices)

# Load the best preforming model
model = torch.load('data/models/ShapeNetDeepB.pth', weights_only=False)

#Generate a new deformation from the same nominal cube (not in train or test dataset, but draw from a similar distribution)
deformed_cube = cube10.deform(rng=np.random.default_rng(9876543210), inplace=False)
print("The deformed and aligned shape's vertices [x,y,z]\n", deformed_cube.vertices)

# Get the LSRF (Approximately Aligned) cube/alignment
best_fit_htm, _, _ = best_fit_transform(deformed_cube.vertices, cube10.vertices)
best_fit_cube = Shape(apply_htm(best_fit_htm, deformed_cube.vertices))
print('The best fit (LSRF) homogenous transform:\n', best_fit_htm)

# Get the weighted fit estimation of the cube/alignment
weighted_htm = weighted_fit(deformed_cube.vertices, cube10.vertices, cube10.variability_matrix(), out_htm=True)
weighted_cube = Shape(apply_htm(weighted_htm, deformed_cube.vertices))
print('The weighted fit homogenous transform:\n', best_fit_htm)

# Get a models estimation of the alignment
model_htm = frombatch(model(tobatch(cube10.deform(inplace=False))))
model_cube = Shape(apply_htm(model_htm, deformed_cube.vertices))
print('The NN fit (ShapeNetDeep B) homogenous transform:\n', best_fit_htm)