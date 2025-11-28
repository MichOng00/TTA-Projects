# 9-layer CNN (from https://github.com/nateraw/quickdraw-pytorch)
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
            nn.Conv2d(1, 64, 3, padding='same'),   # 1x28x28 -> 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 64x28x28 -> 64x14x14

            nn.Conv2d(64, 128, 3, padding='same'), # 64x14x14 -> 128x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                       # 128x14x14 -> 128x7x7

            nn.Conv2d(128, 256, 3, padding='same'),# 128x7x7 -> 256x7x7
            nn.ReLU(),
            nn.MaxPool2d(2), # default ceil_mode=False # 256x7x7 -> 256x3x3

            # fully-connected classifier
            nn.Flatten(),
            nn.Linear(256*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    # Settings
    root = "../QuickDraw"
    class_limit = 10  # train on 10 classes for speed
    max_items_per_class = 2000

    # Prepare dataset
    dataset = QuickDrawDataset(root, max_items_per_class=max_items_per_class, class_limit=class_limit, 
                               class_names=["apple", "The_Eiffel_Tower", "cat", "smiley_face", "sun", "toothbrush", "pizza", "hedgehog", "lighthouse", "ice_cream"])
    # TODO: select class names from https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt
    train_ds, val_ds = dataset.split(pct=0.1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

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
        loop = tqdm(train_loader, desc="Training", leave=False)
        for batch in loop:
            images = batch['pixel_values']
            labels = batch['labels']
            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} done")

    torch.save(model.state_dict(), Path(root) / "quickdraw_model_9layer.pth")
    print("Model and class names saved.")