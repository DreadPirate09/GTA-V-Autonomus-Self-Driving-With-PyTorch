import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

"""

The directory structure should look like this :

data/
    train/
        label_0/
            image0.jpg
            image1.jpg
        label_1/
            image2.jpg
            image3.jpg
    validation/
        label_0/
            image4.jpg
        label_1/
            image5.jpg

"""

# Define directories for train and validation data
train_dir = './all_training_data'
validation_dir = './all_validation_data'
output_dir = './output_data'

# Define transformations to be applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Define a function to save images as .pt files (PyTorch tensor files)
def save_as_tensor(dataset, output_dir, dataset_type='train'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image, label) in enumerate(dataset):
        file_name = f'{dataset_type}_{idx}.pt'
        output_path = os.path.join(output_dir, file_name)
        torch.save((image, label), output_path)
        print(f'Saved {output_path}')

# Step 1: Create PyTorch Dataset for Training
train_dataset = ImageFolder(root=train_dir, transform=transform)
validation_dataset = ImageFolder(root=validation_dir, transform=transform)

# Step 2: Save as .pt files (optional, if you want to save preprocessed data)
save_as_tensor(train_dataset, output_dir, dataset_type='train')
save_as_tensor(validation_dataset, output_dir, dataset_type='validation')

# Step 3: Load the data for training using DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)

# Example of using DataLoader in a training loop
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f" - Images batch size: {images.size()}")
    print(f" - Labels batch size: {labels.size()}")

# Now you can use `train_loader` and `validation_loader` in your PyTorch model training loop.
