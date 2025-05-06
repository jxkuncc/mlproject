"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: April 28, 2025
"""

from copy import deepcopy
from shapelib.Fits import best_fit_transform, apply_htm
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation
from shapelib.Shapes import Shape
import torch
from torch.utils.data import Dataset
from warnings import warn



# def data_from_shape(nominal_shape:Shape, num_samples:int, out_dir:str, out_prefix:str, rng:np.random.Generator = np.random.default_rng())->None:
#     """_summary_
#         Given a nominal shape generate [num_samples] deformed shapes and store 
#         the best fit shape (to the nominal) and the difference from the deformed 
#         shape to best fit
        
#         Outputs Two files (Per Sample) + a listing of paired files:

#         data_dir/out_prefix_[sample#].json -> A sampled deformed shape best fit
#             to the nominal shape
#         data_dir/out_prefix_[sample#].difference -> A csv with the 
#             point-to-point differences of the deformed shape to best-fit shape 
#             (deformed shape - best-fit)

#     Args:
#         nominal_shape (Shape): The nominal shape to sample
#         num_samples (int): The number of samples
#         out_dir (str): The output directory of the generated files
#         out_prefix (str): The output file prefix
#     """
#     assert num_samples >0
#     if num_samples > 1000: 
#         warn('Attempting to generate a large data set (many files). This may take a while...')

#     output_path = Path(out_dir)

#     pairs = []
#     for i in range(num_samples):
#         deformed_shape = Shape(nominal_shape.vertices + nominal_shape.deformation(rng))

#         best_fit = deepcopy(deformed_shape)
#         best_fit.best_fit_to(nominal_shape)

#         best_fit_file = output_path.joinpath(out_prefix+str(i+1)+'.fit')
#         best_fit_df = pd.DataFrame({'x_fit':best_fit.vertices[:,0], 'y_fit':best_fit.vertices[:,1] , 'z_fit':best_fit.vertices[:,2]})
#         best_fit_df.to_csv(best_fit_file, index=False)
     
#         diff = deformed_shape - best_fit
#         diff_file = output_path.joinpath(out_prefix+str(i+1)+'.difference')
#         diff_df = pd.DataFrame({'x_diff':diff[:,0], 'y_diff':diff[:,1], 'z_diff':diff[:,2]})
#         diff_df.to_csv(diff_file, index=False)

#         pairs.append([best_fit_file.name, diff_file.name])
    
#     pairs_file = output_path.joinpath(out_prefix+'.listing')
#     pairs = np.array(pairs)
#     pairs_df = pd.DataFrame({'fit_files':pairs[:,0], 'difference_files':pairs[:,1]})
#     pairs_df.to_csv(pairs_file, index=False)


def data_from_shape(nominal_shape:Shape, num_samples:int, out_dir:str, out_prefix:str, rng:np.random.Generator = np.random.default_rng())->None:
    assert num_samples >0
    if num_samples > 1000: 
        warn('Attempting to generate a large data set (many files). This may take a while...')

    output_path = Path(out_dir)
    pairs = []
    for i in range(num_samples):
        deformed_vertices = nominal_shape.vertices + nominal_shape.deformation(rng)

        bf_htm, _, _ = best_fit_transform(deformed_vertices, nominal_shape.vertices)
        bf_vertices = apply_htm(bf_htm, deformed_vertices)


        best_fit_file = output_path.joinpath(out_prefix+str(i+1)+'.fit')
        best_fit_df = pd.DataFrame({'x':bf_vertices[:,0], 'y':bf_vertices[:,1] , 'z':bf_vertices[:,2]})
        best_fit_df.to_csv(best_fit_file, index=False)

        undo_bf_htm = np.linalg.inv(bf_htm)
        
        transformation_file = output_path.joinpath(out_prefix+str(i+1)+'.transform')
        transformation_df = pd.DataFrame({
            'htm_1':undo_bf_htm[:,0], 
            'htm_2':undo_bf_htm[:,1], 
            'htm_3':undo_bf_htm[:,2], 
            'htm_4':undo_bf_htm[:,3]
            })        
        transformation_df.to_csv(transformation_file, index=False)

        pairs.append([best_fit_file.name, transformation_file.name])
    
    pairs_file = output_path.joinpath(out_prefix+'.listing')
    pairs = np.array(pairs)
    pairs_df = pd.DataFrame({'fit_files':pairs[:,0], 'transform_files':pairs[:,1]})
    pairs_df.to_csv(pairs_file, index=False)


class ShapeDataset(Dataset):
    """A Shape Data Set"""

    def __init__(self, dir:str, listing_file:str, transform=None):
        self.dir = Path(dir)
        self.file_pairs = pd.read_csv(listing_file).to_numpy()
        self.transform = transform

    def __len__(self):
        return self.file_pairs.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise NotImplementedError('__getitem__ encountered a tensor')
            #idx = idx.tolist()
    
        shape = torch.from_numpy(
            pd.read_csv(self.dir.joinpath(self.file_pairs[idx,0])).to_numpy(),
        )
        #shape = torch.reshape(shape, (1, *shape.size()))

        htm = torch.from_numpy(
            pd.read_csv(self.dir.joinpath(self.file_pairs[idx,1])).to_numpy()
        )
        #differences = torch.reshape(differences, (1, *differences.size()))

        if self.transform:
            raise NotImplementedError('__getitem__ tried to transform the data')
            #sample = self.transform(sample)

        return shape, htm





