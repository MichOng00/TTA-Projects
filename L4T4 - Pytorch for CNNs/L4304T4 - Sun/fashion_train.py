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
            nn.Conv2d(1, 16, 3, padding=1),   # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 16x28x28 -> 16x14x14
            
            nn.Conv2d(16, 32, 3, padding=1),  # 16x14x14 -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 32x14x14 -> 32x7x7
        )
        # fully-connected classifier
        # multilayer perceptron
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten to 1D vector
        x = self.fc(x)
        return x
    
def evaluate(model, loader, criterion, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # per-class accuracy counts
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    per_class_acc = {class_names[i] : class_correct[i] / class_total[i]
                     for i in range(len(class_names))}
    return avg_loss, accuracy, per_class_acc

if __name__ == "__main__":
    transform = transforms.ToTensor() # scale pixel values to [0.0, 1.0]
    train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = FashionCNN()
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # track metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    per_class_history = []

    # training loop
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False)
        running_loss = 0
        for images, labels in loop:
            preds = model(images) # forward pass
            loss = criterion(preds, labels) # bigger loss = worse performance

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # update progress bar
            current_loss = loss.item()
            loop.set_postfix(loss=current_loss)

        avg_train_loss = running_loss / len(train_loader)
        test_loss, test_acc, per_class_acc = evaluate(model, test_loader, criterion, class_names)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        per_class_history.append(per_class_acc)

        print(f"Epoch {epoch + 1} done")
        print("Per-class accuracy:")
        for cls, acc in per_class_acc.items():
            print(f"   {cls:12s}: {acc:.4f}")
    torch.save(model.state_dict(), "fashion_model.pth")

    # plot results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")

    plt.tight_layout()
    plt.savefig("fashion_train.png")
    plt.show()