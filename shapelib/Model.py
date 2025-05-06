import torch
from torch import nn

class ShapeNet(nn.Module):
    def __init__(self, num_points:int=8):
        #if num_points<3:raise ValueError('wee need at least three points')
        super().__init__()
        self.layers = torch.nn.Sequential(
            #Number of points times x,y,z total size = numpoints *3
            torch.nn.Flatten(),
            torch.nn.Linear(24, 21, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(21, 18, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(18, 16, dtype=torch.float64), 
            torch.nn.ReLU(), 
            # torch.nn.Linear(15, 12, dtype=torch.float64),
            # torch.nn.ReLU(), 
            # torch.nn.Linear(12, 9, dtype=torch.float64),
            # torch.nn.ReLU(), 
            # torch.nn.Linear(9, 6, dtype=torch.float64) 
        ) 
          



    def forward(self, x):
        #print('forward',x.size())
        output = self.layers(x) 
        output = torch.reshape(output,(output.size()[0], 4,4))
        #print('output',output.size())
        return output 