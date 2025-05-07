# mlproject - Deformation & Alignment Nureal networks to produce aligments of deformed shapes.

A place to store my final project code for MEGR 8090 - Machine Learning in Manufacturing


## An Overview of Files

1) /data/10cube includes all generated test and training data
2) /data/models/ contains all trained NN Models
3) /data/results/ just has RSME values for each model
4) shapelib is a python module with code I wrote to handle all things shapes, fitting, and my NN Models.
  -  shaplib.Data.py handles, generating a dataset from a shapelib.Shape and has an implementation of torch.utils.Dataset to handle shape datasets (shapelib.Data.ShapeDataSet)
  -  shabelib.Fits is a helper submodule with the weighted fit implementation, a best-rigid-fit implementation and a few convenience methods
  -  shabelib.Models containes the torch.nn.Module implementaions of my ShapeNetSimple and ShapeNetDeep NN models and a few convenience methods for using them
  -  shapelib.Shapes contains the logic and classes for handling shapes (reading/writing), defining variabilities, plotting etc.
1) Top level files are files to make shapelib do work
  - data_generation.py is used to generate large-ish datasets for the models
  - deep_trian.py and simple_train.py are used to train ShapeNetDeep and ShapeNetSimple respectively
  - assesment.py is the testing and RSME calculations for the alignment techniques
  - results_visualization.py is a simple script to plot the alignment methods in action
  - example_usage.py is just an example script on how to use all methods 
