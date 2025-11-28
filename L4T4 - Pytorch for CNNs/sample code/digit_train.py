import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(1, 16, 3, padding=1),  # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x28x28 -> 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), # 16x14x14 -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                  # 32x14x14 -> 32x7x7
        )

        # Fully-connected classifier
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),      # flatten 32x7x7 -> 1568 -> 128
            nn.ReLU(),
            nn.Linear(128, 10)               # 128 -> 10 output logits (digits 0–9)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten batch of feature maps to vectors
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # Data preparation
    transform = transforms.ToTensor() # scale pixel values to [0.0, 1.0]
    train_set = datasets.MNIST(root='.', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Instantiate model, loss function, and optimizer
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss() # combines softmax and NLL loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_list = []

    # Training loop
    for epoch in range(3): # train for 3 epochs
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in loop:
            preds = model(images) # forward pass
            loss = criterion(preds, labels) # compute loss

            optimizer.zero_grad()# reset gradients
            loss.backward()
            optimizer.step() # update weights

            # Update progress bar with current loss value
            current_loss = loss.item()
            loss_list.append(current_loss)
            loop.set_postfix(loss=current_loss)

        print(f"Epoch {epoch + 1} done")

    torch.save(model.state_dict(), "digit_model.pth")

    # Plot the loss_list using matplotlib
    total_batches = len(loss_list)
    batch_size = train_loader.batch_size
    x = [i * batch_size for i in range(total_batches)]
    plt.figure(figsize=(10, 5))
    plt.plot(x, loss_list, label="Training Loss")
    plt.xlabel("Number of examples seen")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()


# plot train, test, validation per epoch

# from torch.utils.data import random_split

# if __name__ == "__main__":
#     # Data preparation
#     transform = transforms.ToTensor()
#     full_train_set = datasets.MNIST(root='.', train=True, download=False, transform=transform)
#     test_set = datasets.MNIST(root='.', train=False, download=False, transform=transform)

#     # Split train into train/val
#     train_size = int(0.9 * len(full_train_set))
#     val_size = len(full_train_set) - train_size
#     train_set, val_set = random_split(full_train_set, [train_size, val_size])

#     train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

#     model = DigitCNN()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     train_loss_per_epoch = []
#     val_loss_per_epoch = []
#     test_loss_per_epoch = []

#     for epoch in range(10):
#         print(f"Epoch {epoch + 1}")
#         model.train()
#         running_train_loss = 0.0
#         for images, labels in tqdm(train_loader, desc="Training", leave=False):
#             preds = model(images)
#             loss = criterion(preds, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_train_loss += loss.item() * images.size(0)
#         avg_train_loss = running_train_loss / len(train_loader.dataset)
#         train_loss_per_epoch.append(avg_train_loss)

#         # Validation loss
#         model.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 preds = model(images)
#                 loss = criterion(preds, labels)
#                 running_val_loss += loss.item() * images.size(0)
#         avg_val_loss = running_val_loss / len(val_loader.dataset)
#         val_loss_per_epoch.append(avg_val_loss)

#         # Test loss
#         running_test_loss = 0.0
#         with torch.no_grad():
#             for images, labels in test_loader:
#                 preds = model(images)
#                 loss = criterion(preds, labels)
#                 running_test_loss += loss.item() * images.size(0)
#         avg_test_loss = running_test_loss / len(test_loader.dataset)
#         test_loss_per_epoch.append(avg_test_loss)

#         print(f"Epoch {epoch + 1} done. Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Test loss: {avg_test_loss:.4f}")

#     torch.save(model.state_dict(), "digit_model.pth")

#     # Plot all losses
#     epochs = range(1, len(train_loss_per_epoch) + 1)
#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, train_loss_per_epoch, label="Train Loss", marker='o')
#     plt.plot(epochs, val_loss_per_epoch, label="Validation Loss", marker='o')
#     plt.plot(epochs, test_loss_per_epoch, label="Test Loss", marker='o')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Curves")
#     plt.legend()
#     plt.show()