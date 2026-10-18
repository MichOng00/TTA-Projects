import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(1, 16, 3, padding=1), # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                # 16x28x28 -> 16x14x14
            nn.Conv2d(16, 32, 3, padding=1),# 16x14x14 -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32x14x14 -> 32x7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # digits 0-9
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flattening 32x7x7 into 1-dimensional vector
        x = self.fc(x)
        return x

if __name__ == "__main__": # make sure the code only runs if you run the file directly (not imports)
    # load data
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # create model, loss function, optimiser
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss() # tells AI how well it is doing (bigger loss = worse)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False) # progress bar

        for images, labels in loop:
            preds = model(images) # predict each image using the CNN
            loss = criterion(preds, labels) # calculate difference between predicted and actual labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item()) # show current loss on progress bar

    torch.save(model.state_dict(), "digit_model.pth")