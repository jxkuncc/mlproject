import torch
from torch import nn

class ShapeAutoEncoder(nn.Module):
    def __init__(self, num_points:int=None):
        #if num_points<3:raise ValueError('wee need at least three points')
        super().__init__()
        self.encoder = torch.nn.Sequential(
            #Number of points times x,y,z total size = numpoints *3
            torch.nn.Flatten(),
            torch.nn.Linear(24, 21, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(21, 18, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(18, 15, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(15, 12, dtype=torch.float64),
            torch.nn.ReLU(), 
            torch.nn.Linear(12, 9, dtype=torch.float64),
            torch.nn.ReLU(), 
            torch.nn.Linear(9, 6, dtype=torch.float64) 
        ) 
          
        self.decoder = torch.nn.Sequential( 
            torch.nn.Linear(6, 9, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(9, 12, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(12, 15, dtype=torch.float64), 
            torch.nn.ReLU(), 
            torch.nn.Linear(15, 18, dtype=torch.float64),
            torch.nn.ReLU(), 
            torch.nn.Linear(18, 21, dtype=torch.float64),
            torch.nn.ReLU(), 
            torch.nn.Linear(21, 24, dtype=torch.float64), 
            torch.nn.Sigmoid()
        ) 


    def forward(self, x):
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        decoded =torch.reshape(decoded,(1,8,3))
        return decoded 