"""
This is AlexNet from scratch with random initialisation of weights
Source: Krizhevsky, Sutskever, Hinton -- NeurIPS 2012.
LRN replaced with BatchNorm (standard modern practice).

Input: 224x224x3

Layer   Type        Kernel  Stride  Pad   Output
------  ----------  ------  ------  ----  --------------
Conv1   Conv2d      11x11   4       0     54x54x96
        MaxPool2d   3x3     2       0     26x26x96
Conv2   Conv2d      5x5     1       2     26x26x256
        MaxPool2d   3x3     2       0     12x12x256
Conv3   Conv2d      3x3     1       1     12x12x384
Conv4   Conv2d      3x3     1       1     12x12x384
Conv5   Conv2d      3x3     1       1     12x12x256
        MaxPool2d   3x3     2       0     5x5x256
Flatten                                   6400
FC1     Linear                            4096
FC2     Linear                            4096
FC3     Linear                            500 (num_classes)

Each Conv2d is followed by BatchNorm2d + ReLU (omitted above for space).
Each of FC1/FC2 is preceded by Dropout(p=0.5).
"""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        
        # initialise convolution layers (feature extraction)
        
        self.features = nn.Sequential(
            # Conv Layer 1: 224x224x3 -> 54x54x96 -> pool -> 26x26x96
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Layer 2: 26x26x96 -> 26x26x256 -> pool -> 12x12x256
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Layer 3: 12x12x256 -> 12x12x384
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            # Conv Layer 4: 12x12x384 -> 12x12x384
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            
            # Conv Layer 5: 12x12x384 -> 12x12x256 -> pool -> 5x5x256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # fully connected layers become the classifier
        # dropout will help the model from overfitting if the fully connected layers naturally use all parameters
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        # keep batch dimension, flatten the rest
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Sanity check: confirm every shape along the way matches the diagram,
    # rather than trusting the 256*5*5 hardcoded above blindly.
    model = AlexNet(num_classes=500)

    dummy = torch.zeros(2, 3, 224, 224) 
    feat_out = model.features(dummy)
    print(f"after features: {feat_out.shape}") 

    flat = torch.flatten(feat_out, start_dim=1)
    print(f"after flatten:  {flat.shape}")

    out = model(dummy)
    print(f"final output:   {out.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params:   {n_params:,}")