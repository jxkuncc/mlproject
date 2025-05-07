"""
Author: Jorian Khan
Purpose: This code was written for the MEGR Machine Learning in Manufacturing 
        course in the spring of 2025 at University of North Carolina at 
        Charlotte

Date: May 05, 2025
"""
import torch
from torch import nn

class ShapeNetSimple(nn.Module):
    def __init__(self, num_vertices):
        if num_vertices<3:raise ValueError('At least three points need')
        super().__init__()

        # There is a direct linear relationship between the points as given and the translation
        self.translation = nn.Sequential(
            nn.Flatten(1,2),
            nn.Linear(num_vertices*3, 3, dtype=torch.float64),
            nn.Unflatten(1,(3,1)),
            nn.ConstantPad2d((0,0,0,1),1),
            nn.ZeroPad2d((3,0,0,0))
        )

        self.rotation = nn.Sequential(
            nn.Flatten(1,2),
            nn.Linear(num_vertices*3, 9, dtype=torch.float64),
            nn.Tanh(), # to keep values between -1 and 1 i.e. sin(angle) and cos(angle) in rotation matrix
            nn.Unflatten(1,(3,3)),
            nn.ZeroPad2d((0,1,0,1))
        )

    def forward(self, x):
        return self.rotation(x) + self.translation(x)

class ShapeNetDeep(nn.Module):
    def __init__(self, num_vertices:int=8):
        if num_vertices<3:raise ValueError('At least three points need')
        super().__init__()

        # There is a direct linear relationship between the points as given and the translation
        self.translation = nn.Sequential(nn.Flatten(1,2))
        self.translation.extend(ShapeNetDeep._get_layers(num_vertices, target_size=3))
        self.translation.append(nn.Unflatten(1,(3,1)))
        self.translation.append(nn.ConstantPad2d((0,0,0,1),1))
        self.translation.append(nn.ZeroPad2d((3,0,0,0)))

        self.rotation = nn.Sequential(nn.Flatten(1,2))
        self.rotation.extend(ShapeNetDeep._get_layers(num_vertices, target_size=9))
        self.rotation.append(nn.Tanh()) # to keep values between -1 and 1 i.e. sin(angle) and cos(angle) in rotation matrix
        self.rotation.append(nn.Unflatten(1,(3,3)))
        self.rotation.append(nn.ZeroPad2d((0,1,0,1)))

    def forward(self, x):
        return self.rotation(x) + self.translation(x)
    
    @classmethod
    def _get_layers(cls, num_vertices:int, target_size:int = 3):
        if target_size%3 > 0: raise ValueError('target must be a multiple of 3')

        if num_vertices ==3:
            return nn.Sequential(nn.Linear(3, target_size, dtype=torch.float64))

        linear_layers = nn.Sequential()
        remaining = num_vertices*3
        remaining = num_vertices*3
        while remaining !=target_size:
            start_value = remaining
            end_value = remaining -3
            linear_layers.append(nn.Linear(start_value, end_value, dtype=torch.float64))
            remaining-=3

        return linear_layers