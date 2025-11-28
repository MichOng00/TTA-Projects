import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# FashionMNIST class names
class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy counts
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    per_class_acc = {class_names[i]: class_correct[i] / class_total[i]
                     for i in range(len(class_names))}
    return avg_loss, accuracy, per_class_acc

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Instantiate model, loss function, and optimizer
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Track metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    per_class_history = []

    # Training loop
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        model.train()
        loop = tqdm(train_loader, desc="Training", leave=False)
        running_loss = 0

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        test_loss, test_acc, per_class_acc = evaluate(model, test_loader, criterion, device, class_names)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        per_class_history.append(per_class_acc)

        print(f"Epoch {epoch + 1} done | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")
        print("Per-class accuracy:")
        for cls, acc in per_class_acc.items():
            print(f"  {cls:12s}: {acc:.4f}")

    torch.save(model.state_dict(), "fashion_model.pth")

    # Plot results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.tight_layout()
    plt.savefig("fashion_train.png")
    plt.show()
