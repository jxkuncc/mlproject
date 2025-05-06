from time import time
import torch
from torch import nn
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from shapelib.Model import ShapeNet
from torch.utils.data import DataLoader



# Initialize the model
model = ShapeNet()
 
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train the autoencoder
num_epochs = 100

for epoch in range(num_epochs):
    epoch_time = time()
    for data in train_loader:
        fit, transform = data
        #print(fit.size())
        #print(difference.size())
        
        # The result the model should produce
        transform = transform.to(device)

        # The data we feed to the model
        fit = fit.to(device)

        #print(fit[0])
        optimizer.zero_grad()
        # The output from the model
        output = model(fit)
        #print(output[0])

        
        loss = criterion(output, transform)
        loss.backward()
        optimizer.step()
        #break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {time()-epoch_time:0.1f}s')


torch.save(model, 'first_model.pth')