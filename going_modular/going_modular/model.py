"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 

import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    input_shape: An integer indicating the number of input channels.
    hidden_units: An integer indicating the number of hidden units between layers.
    output_shape: An integer indicating the number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=0),  
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Dynamically compute the in_features for the Linear layer
        self._conv_output_size = self._get_conv_output_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self._conv_output_size, out_features=output_shape)
        )

    def _get_conv_output_size(self, input_shape: int) -> int:
        """Calculates the output size after the convolutional layers."""
        with torch.no_grad():
            sample_input = torch.zeros(1, input_shape, 224, 224)  # Assuming 224x224 input size
            sample_output = self.conv_block_1(sample_input)
            sample_output = self.conv_block_2(sample_output)
            return int(torch.prod(torch.tensor(sample_output.shape[1:])))
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

