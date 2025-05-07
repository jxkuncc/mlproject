import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from shapelib.Shapes import Shape
from shapelib.Data import ShapeDataset
from shapelib.Model import ShapeNetSimple, ShapeNetDeep


num_points = 8
t1 = nn.Flatten(1,2)
t2 = nn.Linear(24, 3, dtype=torch.float64)
t3= nn.Unflatten(1,(3,1))
t4 = nn.ZeroPad2d((3,0,0,1))

#t3 = nn.ReLU()


r1 = nn.Flatten(1,2)
r2 = nn.Linear(24, 9, dtype=torch.float64)
r3 = nn.Tanh() # to keep values between -1 and 1 i.e. sin(angle) and cos(angle) in rotation matrix
r4 = nn.Unflatten(1,(3,3))
r5 = nn.ZeroPad2d((0,1,0,1))


# Load dataset
train_dir = 'data/10cube/train'
train_listing = train_dir+'/10cube_train.listing'
train_dataset = ShapeDataset(train_dir, train_listing)

# Define the dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=25, shuffle=True)

#model = ShapeNetSimple()

model = ShapeNetDeep()

for data in train_loader:
    fit = data[0]
    transform = data[1]
    # #print(fit.size())
    # #print(fit[0])
    # #print('---------------------------')
    # t1_out = t1(fit)
    # #print('t1_out',t1_out.size())
    # #print(t1_out[0],'\n-----------------')
    # t2_out = t2(t1_out)
    # #print('t2_out',t2_out.size())
    # #print(t2_out[0],'\n-----------------')
    # t3_out = t3(t2_out)
    # #print('t3_out',t3_out.size())
    # #print(t3_out[0],'\n-----------------')
    # t4_out = t4(t3_out)
    # #print('t4_out',t4_out.size())
    # #print(t4_out[0],'\n-----------------')
    
    # r1_out = r1(fit)
    # #print('r1_out',r1_out.size())
    # #print(r1_out[0],'\n-----------------')
    # r2_out = r2(r1_out)
    # #print('r2_out',r2_out.size())
    # #print(r2_out[0],'\n-----------------')
    # r3_out = r3(r2_out)
    # #print('r3_out', r3_out.size())
    # #print(r3_out[0],'\n-----------------')
    # r4_out = r4(r3_out)
    # #print('r4_out', r4_out.size())
    # #print(r4_out[0],'\n-----------------')
    # r5_out = r5(r4_out)
    # #print('r5_out', r5_out.size())
    # #print(r5_out[0],'\n-----------------')
    

    # #print('---------------------------')

    
    # base = torch.zeros((25, 4, 4), dtype=torch.float64)
    # base[:,-1,-1] = 1
    # #print('base', base[0])

    # final = base + t4_out + r5_out
    # print('Final\n', final[0]) 


    print('test\n',model(fit)[0])

    

    #print(transform)

    break