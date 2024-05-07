import os
import timm
import torch
import torch.nn as nn
from utils import TARGETS_TO_CLASSES, MODEL_DIMS, get_model
from torchvision import models


NUM_TARGETS = 13


class MedicalParametersModel(torch.nn.Module):
    def __init__(self, two_layers, hidden_dim, is_regression,
                 model_name, target, unfreeze, from_scratch):
        super().__init__()
        self.two_layers = two_layers
        
        self.model = get_model(model_name, unfreeze, from_scratch)

        if is_regression:
            output_dim = (NUM_TARGETS if target == 'all' else 1)
        else:
            output_dim = 2

        if two_layers:
            self.linear1 = nn.Linear(MODEL_DIMS[model_name], hidden_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, output_dim)
        else:
            self.linear = nn.Linear(MODEL_DIMS[model_name], output_dim)


    def forward(self, x):
        if self.two_layers:
            return self.linear2(self.relu(self.linear1(self.model(x))))
        return self.linear(self.model(x))

            