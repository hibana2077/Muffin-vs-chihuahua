'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-20 15:37:01
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-20 15:43:48
FilePath: \Muffin-vs-chihuahua\train\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE/
'''
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import timm
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# load data 
train_data_dir = '../data/archive/train'
test_data_dir = '../data/archive/test'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(train_data_dir, transform=train_transform)
test_dataset = ImageFolder(test_data_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

dummy_input = torch.randn(1, 3, 224, 224)

# define model

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = timm.create_model('vit_small_resnet26d_224', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, 2)

    def forward(self, x):
        return self.model(x)
    
m = Model()
out = m(dummy_input)
print(out.shape)

x,y = next(iter(train_loader))
print(f"input shape: {x.shape}, output shape: {m(x).shape}, label shape: {y.shape}")

# define loss and optimizer

import timm.optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = m.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = timm.optim.Lookahead(timm.optim.RAdam(m.parameters(), lr=1e-3))
LR_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# train model
loss_history = []
EPOCHS = 6

for EPOCH in range(EPOCHS):
    m.train()
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = m(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    LR_scheduler.step()
    
    print(f"Epoch {EPOCH+1}, Loss: {running_loss/len(train_loader)}, LR: {optimizer.param_groups[0]['lr']}")
    loss_history.append(running_loss/len(train_loader))

# plot loss history
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('../img/loss.png')

# test output

m.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = m(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100*correct/total}%")

# random show image

import matplotlib.pyplot as plt
import numpy as np
import torchvision

random_idx = np.random.randint(0, len(test_dataset), 1)[0]
image, label = test_dataset[random_idx]
image = image.unsqueeze(0).to(device)

# predict
m.eval()
with torch.no_grad():
    output = m(image)
    _, predicted = torch.max(output.data, 1)


plt.imshow(image.squeeze().cpu().numpy().transpose(1,2,0))
plt.title(f"Predicted: {predicted.item()}, True: {label}")
plt.savefig('../img/predict.png')