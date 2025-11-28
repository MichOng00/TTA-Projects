import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional feature extractor
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(1, 16, 3, padding=1), # 1x28x28 -> 16x28x28
            nn.ReLU(), 
            nn.MaxPool2d(2),                # 16x28x28 -> 16x14x14
            
            nn.Conv2d(16, 32, 3, padding=1),# 16x14x14 -> 32x14x14
            nn.ReLU(), 
            nn.MaxPool2d(2)                 # 32x14x14 -> 32x7x7
        )
        # fully-connected
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # digits 0-9
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten batch of feature maps to vectors
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    transform = transforms.ToTensor()

    train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        model.train()
        loop = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in loop:
            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss = loss.item())

        print(f"Epoch {epoch + 1} done")

    torch.save(model.state_dict(), "fashion_model.pth")