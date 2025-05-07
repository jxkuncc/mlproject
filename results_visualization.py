"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: May 7, 2025
"""

import numpy as np
import pyvista as pv
from shapelib.Shapes import Shape, polycube
from shapelib.Fits import weighted_fit, best_fit_transform, apply_htm
from shapelib.Models import tobatch, frombatch
import torch




# Load the nominal shape
cube10 = Shape.from_file('data/10cube/10cube.json')

# Load the best preforming model
model = torch.load('data/models/ShapeNetDeepB.pth', weights_only=False)

#Generate a new deformation from the same nominal cube (not in train or test dataset, but draw from a similar distribution)
deformed_cube = cube10.deform(rng=np.random.default_rng(9876543210), inplace=False)

# Get the LSRF (Approximately Aligned) cube/alignment
best_fit_htm, _, _ = best_fit_transform(deformed_cube.vertices, cube10.vertices)
best_fit_cube = Shape(apply_htm(best_fit_htm, deformed_cube.vertices))

# Get the weighted fit estimation of the cube/alignment
weighted_htm = weighted_fit(deformed_cube.vertices, cube10.vertices, cube10.variability_matrix(), out_htm=True)
weighted_cube = Shape(apply_htm(weighted_htm, deformed_cube.vertices))

# Get a models estimation of the alignment
model_htm = frombatch(model(tobatch(cube10.deform(inplace=False))))
model_cube = Shape(apply_htm(model_htm, deformed_cube.vertices))


#Plot the alignments

plt = pv.Plotter(shape=(1, 4))

def lines(vertsa, vertsb, **kwargs):
    plt.add_lines(np.array(list(zip(vertsa, vertsb))).reshape((-1,3)), **kwargs)

plt.subplot(0,0)
plt.add_title('Correct Alignment (Deformed)',font_size=12)
plt.add_mesh(polycube(cube10), color='k', style='wireframe', point_size=3, opacity=0.2)
plt.add_mesh(polycube(deformed_cube), opacity=0.1,color='g')
plt.add_mesh(polycube(deformed_cube), color='g', style='wireframe', show_vertices=True, point_size=3)

plt.subplot(0,1)
plt.add_title('Approximate Alignment (LSRF)',font_size=12)
plt.add_mesh(polycube(cube10), color='k', style='wireframe', point_size=3, opacity=0.2)
plt.add_mesh(polycube(deformed_cube), opacity=0.07,color='g')
plt.add_mesh(polycube(deformed_cube), color='g', style='wireframe', show_vertices=True, point_size=3)
plt.add_mesh(polycube(best_fit_cube), opacity=0.1,color='tab:orange')
plt.add_mesh(polycube(best_fit_cube), color='tab:orange', style='wireframe', show_vertices=True, point_size=3)
lines(deformed_cube.vertices, best_fit_cube.vertices, color='r', width = 3)

plt.subplot(0,2)
plt.add_title('Weighted Fit Alignment',font_size=12)
plt.add_mesh(polycube(cube10), color='k', style='wireframe', point_size=3, opacity=0.2)
plt.add_mesh(polycube(deformed_cube), opacity=0.07,color='g')
plt.add_mesh(polycube(deformed_cube), color='g', style='wireframe', show_vertices=True, point_size=3)
plt.add_mesh(polycube(weighted_cube), opacity=0.1,color='m')
plt.add_mesh(polycube(weighted_cube), color='m', style='wireframe', show_vertices=True, point_size=3)
lines(deformed_cube.vertices, weighted_cube.vertices, color='r', width = 3)

plt.subplot(0,3)
plt.add_title('ShapeNetDeep B Alignment',font_size=12)
plt.add_mesh(polycube(cube10), color='k', style='wireframe', point_size=3, opacity=0.2)
plt.add_mesh(polycube(deformed_cube), opacity=0.07, color='g')
plt.add_mesh(polycube(deformed_cube), color='g', style='wireframe', show_vertices=True, point_size=3)
plt.add_mesh(polycube(model_cube), opacity=0.1, color='b')
plt.add_mesh(polycube(model_cube), color='b', style='wireframe', show_vertices=True, point_size=3)
lines(deformed_cube.vertices, model_cube.vertices, color='r', width = 3)

plt.link_views()
plt.camera_position = 'xz'
plt.camera.elevation = 30
plt.camera.azimuth = 30
plt.show()