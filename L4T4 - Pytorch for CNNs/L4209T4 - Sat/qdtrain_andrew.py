import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from quickdraw_utils import QuickDrawDataSet

class QuickDrawCNN9L(nn.Module):
    def __init__(self,num_classes):
        super(). __init__()
        self.model=nn.Sequential(
            nn.Conv2d(1, 64, 3, padding="same"), # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding="same"), # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding="same"), # 1x28x28 -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(256*3*3,512),
            nn.ReLU(),
            nn.Linear(512,num_classes)                     
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__=="__main__":
    root="QuickDraw"
    class_limit=10
    max_items_per_class=2000

    dataset=QuickDrawDataSet(root,max_items_per_class,class_limit,
                             class_names=["airplane","banana","bicycle","cactus","camera","cat","coffee_cup","guitar","lighthouse","umbrella"])
    
    train_ds, val_ds=dataset.split(pct=0.1)
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True,collate_fn=dataset.collate_fn)
    val_loader=DataLoader(val_ds,batch_size=64,shuffle=False,collate_fn=dataset.collate_fn)


    class_names=dataset.classes
    Path(root).mkdir(exist_ok=True)
    with open(Path(root)/"class_names.txt","w") as f:
        f.write("\n".join(class_names))

    model=QuickDrawCNN9L(num_classes=len(class_names))
    criterion=nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(3):
        print(f"Epoch:{epoch + 1}")
        loop=tqdm(train_loader,desc="Training",leave=False)
        for batch in loop:
            images=batch['pixel_values']
            labels=batch['labels']
            preds=model(images)
            loss=criterion(preds,labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch{epoch+1}done.")
    torch.save(model.state_dict(),Path(root)/"quickdraw_9layer.pth")
