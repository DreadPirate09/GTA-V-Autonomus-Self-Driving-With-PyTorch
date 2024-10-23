import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from gta_v_driver_model import GTAVDriverModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support


# Define a custom dataset
class GTAVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the steering_angle, throttle, brake, speed, and image path
        steering_angle = self.data.iloc[idx, 0]
        throttle = self.data.iloc[idx, 1]
        brake = self.data.iloc[idx, 2]
        speed = self.data.iloc[idx, 3]
        img_path = self.data.iloc[idx, 4]
        
        # Open the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply any transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Combine steering_angle, throttle, brake into a tensor (labels)
        labels = torch.tensor([steering_angle, throttle, brake], dtype=torch.float32)
        
        # Return the speed as the first feature and flattened image data as the second
        features = torch.cat([torch.tensor([speed], dtype=torch.float32), image.flatten()])
        
        return features, labels

# Define the transformation for the images (resize to 160x640 and normalize)
transform = transforms.Compose([
    transforms.Resize((160, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
csv_file = 'data/data.csv'
img_dir = 'data/'

# Create the dataset and dataloader
dataset = GTAVDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GTAVDriverModel().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error loss for regression
optimizer = optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter(log_dir='runs/gtav_driver')  # Logs will be saved here

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            avg_loss = running_loss / 10
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            writer.add_scalar('training_loss', avg_loss, epoch * len(dataloader) + i)  # Log to TensorBoard
            running_loss = 0.0

    # Optional: log model weights, histograms, or images
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'gtav_driver_model.pth')


# Close TensorBoard writer
writer.close()