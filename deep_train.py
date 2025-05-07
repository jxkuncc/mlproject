"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: May 05, 2025
"""
from shapelib.Data import ShapeDataset
from shapelib.Models import ShapeNetDeep
from time import time
import torch
from torch import nn
from torch.utils.data import DataLoader



# Initialize the model
model = ShapeNetDeep(num_vertices=8)
 
# Load dataset
train_dir = 'data/10cube/train'
train_listing = train_dir+'/10cube_train.listing'
train_dataset = ShapeDataset(train_dir, train_listing)

# Define the dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=25, shuffle=True)

# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}...')
model.to(device)
 
# Define the loss function and optimizer
criterion = nn.MSELoss() #nn.L1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Train the Network
num_epochs = 100
for epoch in range(num_epochs):
    epoch_time = time()
    for data in train_loader:
        fit, transform = data

        # The result the model should produce
        transform = transform.to(device)

        # The data we feed to the model
        fit = fit.to(device)
 
        optimizer.zero_grad()
 
        # The output from the model
        output = model(fit)

        # Compute the loss and back-prop
        loss = criterion(output, transform)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {time()-epoch_time:0.1f}s')


# torch.save(model, 'data/models/ShapeNetDeepL1LossA.pth')
torch.save(model, 'data/models/ShapeNetDeepMSEA.pth')