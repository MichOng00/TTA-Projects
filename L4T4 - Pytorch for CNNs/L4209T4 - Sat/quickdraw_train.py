import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from quickdraw_utils import QuickDrawDataset

class QuickDrawCNN9L(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # create a 9-layer CNN
            # the convolution layers have 64, 128, and 256 output channels.

            # nn.Conv2d(input_channels, output_channels, kernel_size, padding)
            nn.Conv2d(1, 64, 3, padding="same"), # 1x28x28 -> 64x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 64x28x28 -> 64x14x14
            
            nn.Conv2d(64, 128, 3, padding="same"), # 64x14x14 -> 128x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 128x14x14 -> 128x7x7

            nn.Conv2d(128, 256, 3, padding="same"), # 128x7x7 -> 256x7x7
            nn.ReLU(),
            nn.MaxPool2d(2),                     # 256x7x7 -> 256x3x3 - cut off decimal point

            nn.Flatten(),

            # fully-connected
            nn.Linear(256*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)

        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    root = "../QuickDraw"
    class_limit = 10
    max_items_per_class = 2000

    dataset = QuickDrawDataset(root, max_items_per_class, class_limit, 
                               class_names = ["apple", "The_Eiffel_Tower", "cat", "smiley_face", "sun", "toothbrush", "pizza", "hedgehog", "lighthouse", "ice_cream"])
    train_ds, val_ds = dataset.split(pct=0.1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

    # save class names in a text file on the computer
    class_names = dataset.classes
    Path(root).mkdir(exist_ok=True) # create folder if it does not exist
    with open(Path(root) / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # instantiate model, loss, optimiser
    model = QuickDrawCNN9L(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(3):
        print(f"Epoch: {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False) # progress bar
        for batch in loop:
            images = batch['pixel_values']
            labels = batch['labels'] # actual labels (ground truth)
            preds = model(images)    # predicted labels (what the neural network thinks it is)
            loss = criterion(preds, labels)  # how far the predictions are from the ground truth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1} done.")

    # save the model
    torch.save(model.state_dict(), Path(root) / "quickdraw_9layer.pth")

