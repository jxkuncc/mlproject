import torch
from torch import nn
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from shapelib.Model import ShapeAutoEncoder
from torch.utils.data import Dataset, DataLoader

# dir = 'data/10cube/train'
# listing = dir+'/10cube_train.listing'

# training_set = ShapeDataset(dir, listing)

# train_loader = DataLoader(training_set, batch_size=20, shuffle=True)


# Initialize the autoencoder
model = ShapeAutoEncoder()

 
# Load dataset

train_dir = 'data/10cube/train'
train_listing = train_dir+'/10cube_train.listing'
train_dataset = ShapeDataset(train_dir, train_listing)

test_dir = 'data/10cube/test'
test_listing = test_dir+'/10cube_test.listing'
test_dataset = ShapeDataset(test_dir, test_listing)


# Define the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=1, 
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=1)
 
# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
 
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Train the autoencoder
num_epochs = 1
for epoch in range(num_epochs):
    data_c = 1
    for data in train_loader:
        fit, difference = data
        # print(fit.size())
        # print(difference.size())
        
        difference = difference.to(device)
        fit = fit.to(device)
        optimizer.zero_grad()

        output = model(fit)
        # print(output.size())

        loss = criterion(output, difference)
        loss.backward()
        optimizer.step()
        print(f'Data [{data_c+1}/{5000}], Loss: {loss.item():.4f}')
        data_c+=1

