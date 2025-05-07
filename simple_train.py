from time import time
import torch
from torch import nn
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from shapelib.Model import ShapeNetSimple
from torch.utils.data import DataLoader



# Initialize the model
model = ShapeNetSimple(num_vertices=8)
 
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
criterion = nn.L1Loss()#nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
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


torch.save(model, 'ShapeNetSimpleL1Loss1.pth')