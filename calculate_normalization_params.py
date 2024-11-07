import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import threading
import time

def loading_message():
    while not done:
        print("Loading, compute on a big dataset...", end="\r")
        time.sleep(0.5)
        print("Loading, compute on a big dataset.. ", end="\r")
        time.sleep(0.5)
        print("Loading, compute on a big dataset. ", end="\r")
        time.sleep(0.5)
        print("Loading, compute on a big dataset    ", end="\r")
        time.sleep(0.5)

done = False
thread = threading.Thread(target=loading_message)
thread.start()


class CustomDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

path_dir = 'data'

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(root_dir=path_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

def calculate_mean_std(dataloader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images in dataloader:
        batch_images_count = images.size(0)
        images = images.view(batch_images_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count

    mean /= total_images_count
    std /= total_images_count
    return mean, std

mean, std = calculate_mean_std(dataloader)

done = True
thread.join()
print("Normalizations params calculated successfully")
print("Mean:", mean)
print("Std:", std)
