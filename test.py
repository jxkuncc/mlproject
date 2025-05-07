from time import time
import torch
from torch import nn
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from torch.utils.data import DataLoader




# Load dataset
test_dir = 'data/10cube/test'
test_listing = test_dir+'/10cube_test.listing'
test_dataset = ShapeDataset(test_dir, test_listing)

# Define the dataloader
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

#get the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}...')

#Load the model and move it to the device
# model = torch.load('ShapeNetSimpleMSE1.pth', weights_only=False)
model = torch.load('ShapeNetSimpleL1Loss1.pth', weights_only=False)
model.to(device)



points = test_dataset[0][0]
points = points.expand((1,*points.size()))
points = points.to(device)

actual = test_dataset[0][1]
print(actual)


pred = model(points)
print(pred[0])

# count = 0
# for data in test_loader:
#     fit, transforom = data
#     #print(fit.size())
#     #print(difference.size())
    
#     # The result the model should produce
#     transforom = transforom.to(device)

#     # The data we feed to the model
#     fit = fit.to(device)


#     output = model(fit)
#     print(output)

#     count+=1

#     if count>3: break