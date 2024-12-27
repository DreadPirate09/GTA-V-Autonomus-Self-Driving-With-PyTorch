import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from gta_v_driver_model import GTAVDriverModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from calculate_normalization_params import mean,std

class GTAVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        steering_angle = self.data.iloc[idx, 0]
        throttle = 1.0 - self.data.iloc[idx, 1]
        brake = self.data.iloc[idx, 2]
        speed = self.data.iloc[idx, 3]
        img_path = self.data.iloc[idx, 4]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # labels = torch.tensor([steering_angle, throttle, brake], dtype=torch.float32)
        labels = torch.tensor([steering_angle, throttle], dtype=torch.float32)

        features = torch.cat([torch.tensor([speed], dtype=torch.float32), image.flatten()])
        
        return features, labels

transform = transforms.Compose([
    transforms.Resize((160, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

csv_file = 'data/data.csv'

dataset = GTAVDataset(csv_file=csv_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GTAVDriverModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

writer = SummaryWriter(log_dir='runs/gtav_driver2')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9: 
            avg_loss = running_loss / 10
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('training_loss', avg_loss, epoch * len(dataloader) + i)
            running_loss = 0.0

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

print('Finished Training')

torch.save(model.state_dict(), 'gtav_driver_model.pth')

writer.close()