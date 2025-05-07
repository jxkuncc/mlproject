"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: May 05, 2025
"""
import numpy as np
import pandas as pd
from shapelib.Fits import apply_htm, weighted_fit
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from shapelib.Models import ShapeNetSimple, ShapeNetDeep
from sklearn.metrics import root_mean_squared_error as rmse
from time import time
import torch
from torch.utils.data import DataLoader


def residuals(A, B):
        return np.linalg.norm(A-B, ord = 2, axis=1)

# Load the nominal shape
cube10 = Shape.from_file('data/10cube/10cube.json')
cube10_var = cube10.variability_matrix() # the variability in the xyz positions for the cube for the weighted fit

# Load dataset
test_dir = 'data/10cube/test'
test_listing = test_dir+'/10cube_test.listing'
test_dataset = ShapeDataset(test_dir, test_listing)

# Define the dataloader
test_loader = DataLoader(dataset=test_dataset, batch_size=1) #Size one because we need to handle every residual

#Use the cpu because we need numpy
#device = #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using {device}...')


models = {
    'ShapeNetSimple A':'data/models/ShapeNetSimpleA.pth',
    'ShapeNetSimple B':'data/models/ShapeNetSimpleB.pth',
    'ShapeNetSimple C':'data/models/ShapeNetSimpleC.pth',
    'ShapeNetSimple D':'data/models/ShapeNetSimpleD.pth',
    'ShapeNetSimple E':'data/models/ShapeNetSimpleE.pth',
    'ShapeNetSimple F':'data/models/ShapeNetSimpleF.pth',
    'ShapeNetDeep A':'data/models/ShapeNetDeepA.pth',
    'ShapeNetDeep B':'data/models/ShapeNetDeepB.pth',
    'ShapeNetDeep C':'data/models/ShapeNetDeepC.pth',
    'ShapeNetDeep D':'data/models/ShapeNetDeepD.pth',
    'ShapeNetDeep E':'data/models/ShapeNetDeepE.pth',
    'ShapeNetDeep F':'data/models/ShapeNetDeepF.pth'
}


assessment_time = time()
print('Beginning Assessment...')
results_df = pd.DataFrame(columns=['Model/Approach', 'RMSE Residual'])
finished_weighted = False
weighted_resids = [] 
for model_name, model_path in models.items():
        model_time = time()
        model = torch.load(model_path, weights_only=False)
        model_resids = [] 
        for data in test_loader:
                aprox_aligned = data[0]
                aprox_aligned_np = data[0].numpy()[0]

                actual_htm = data[1].numpy()[0] # The htm taking the aprox. aligned shape to the aligned shape
                actual = apply_htm(actual_htm, aprox_aligned_np) # The aligned deformed shape


                model_htm = model(aprox_aligned) 
                model_htm = model_htm[0].detach().numpy()

                model_pred = apply_htm(model_htm, aprox_aligned_np)
                
                #Store the root-mean-square error for later
                model_resids.append(residuals(actual, model_pred))
                
               
                #We only need to handle the weighted fit once
                if not finished_weighted:
                        weighted_htm = weighted_fit(aprox_aligned_np, cube10.vertices, cube10_var, out_htm=True)
                        weighted_pred = apply_htm(weighted_htm, aprox_aligned_np)
                        weighted_resids.append(residuals(actual, weighted_pred))
                        finished_weighted = True
        
        model_resids = np.array(model_resids).flatten()
        model_rmse_resid = rmse(np.zeros_like(model_resids), model_resids)
        results_df.loc[len(results_df)] = {'Model/Approach':model_name, 'RMSE Residual':model_rmse_resid}
        print(f"Assessment of '{model_name}' completed in {time()-model_time:0.2f}s. RMSE Residual: {model_rmse_resid:0.3f}")

weighted_resids = np.array(weighted_resids).flatten()
weighted_rmse_resid = rmse(np.zeros_like(weighted_resids), weighted_resids)
results_df.loc[len(results_df)] = {'Model/Approach':'Weighted Fit', 'RMSE Residual':weighted_rmse_resid}
print(f"Assessment of 'weighted fit' completed. RMSE Residual: {weighted_rmse_resid:0.3f}")
print(f'Assessment completed in {time()-assessment_time:0.2f}s.')

print(results_df)
results_df.to_csv('data/results/results.csv', index=False)


