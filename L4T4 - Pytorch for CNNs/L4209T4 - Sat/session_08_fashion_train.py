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

def evaluate(model, loader, criterion, class_names):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() # how badly the model is doing

            _, predicted = outputs.max(1) # get the top prediction
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # calculate accuracy per class
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

if __name__ == "__main__": # make sure the code only runs if you run the file directly (not imports)
    # load data
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # create model, loss function, optimiser
    model = FashionCNN()
    criterion = nn.CrossEntropyLoss() # tells AI how well it is doing (bigger loss = worse)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # track metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    per_class_history = []

    # training loop
    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, desc="Training", leave=False) # progress bar
        running_loss = 0

        for images, labels in loop:
            preds = model(images) # predict each image using the CNN
            loss = criterion(preds, labels) # calculate difference between predicted and actual labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_postfix(loss = loss.item()) # show current loss on progress bar

        avg_train_loss = running_loss / len(train_loader)
        test_loss, test_acc, per_class_acc = evaluate(model, test_loader, criterion, class_names)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        per_class_history.append(per_class_acc)

        print(f"Epoch {epoch + 1} done")
        print("Per-class accuracy:")
        for cls, acc in per_class_acc.items():
            print(f"      {cls:12s}: {acc:.4f}")

    torch.save(model.state_dict(), "fashion_model.pth")