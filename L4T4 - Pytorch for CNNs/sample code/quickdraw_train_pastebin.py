import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Import QuickDrawDataset from utils
from quickdraw_utils import QuickDrawDataset

class QuickDrawCNN9L(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # Create a 9-layer CNN (3 convolution layers).
            # The convolution layers have 64, 128, and 256 output channels respectively.
            # You can change these if you want to try and improve performance.

            # TODO: YOUR CODE HERE
            
            nn.Flatten(),

            # Create the fully-connected classifier, with 2 linear layers.
            # The first linear layer has output size 512.
            # What should the number of outputs be for the final linear layer?
            
            # TODO: YOUR CODE HERE
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    # Settings
    root = "../QuickDraw"
    class_limit = 10
    max_items_per_class = 2000

    # Prepare dataset
    # TODO: select 10 class names from https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt
    dataset = QuickDrawDataset(root, max_items_per_class=max_items_per_class, class_limit=class_limit, 
                               class_names=[])
    
    # Split into train and test set

    # Save class names for later use in the app

    # Instantiate model, loss, optimizer

    # Training loop

    # TODO: YOUR CODE HERE
    # Hint: in your inner for loop, use the following to access the items in the dictionary.
    # images = batch['pixel_values']
    # labels = batch['labels']