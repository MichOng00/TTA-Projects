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

            nn.Conv2d(1, 64, 3, padding=1), # 1x28x28 -> 64x28x28
            nn.ReLU(), 
            nn.MaxPool2d(2),                # 64x28x28 -> 64x14x14
            
            nn.Conv2d(64, 128, 3, padding=1),# 64x14x14 -> 128x14x14
            nn.ReLU(), 
            nn.MaxPool2d(2),                 # 128x14x14 -> 128x7x7

            nn.Conv2d(128, 256, 3, padding=1),# 128x7x7 -> 256x7x7
            nn.ReLU(), 
            nn.MaxPool2d(2),                  # 256x7x7 -> 256x3x3
            
            nn.Flatten(),

            # Create the fully-connected classifier, with 2 linear layers.
            # The first linear layer has output size 512.
            # What should the number of outputs be for the final linear layer?
            
            nn.Linear(256*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
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
                               class_names=["apple", "The_Eiffel_Tower", "cat", "smiley_face", "sun", "toothbrush", "pizza", "hedgehog", "lighthouse", "ice_cream"])
    
    # Split into train and test set
    train_ds, val_ds = dataset.split(pct=0.1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
    # don't set suffer=True

    # Save class names for later use in the app
    class_names = dataset.classes
    Path(root).mkdir(exist_ok=True)
    with open(Path(root) / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # Instantiate model, loss, optimizer
    model = QuickDrawCNN9L(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop

    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False) # progress bar

        for batch in loop:
            images = batch["pixel_values"]
            labels = batch["labels"]
            preds = model(images) # predict each image using the CNN
            loss = criterion(preds, labels) # calculate difference between predictions and actual labels

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item())

    torch.save(model.state_dict(), Path(root) / "quickdraw_model.pth")