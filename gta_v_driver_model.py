import torch
import torch.nn as nn
import torch.nn.functional as F

class GTAVDriverModel(nn.Module):
    def __init__(self, height=160, width=640):
        super(GTAVDriverModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=0)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv5 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv7 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=0)
        
        # Fully connected layers (dense layers)
        # You need to calculate the size of the flattened conv8 output based on input dimensions
        flattened_size = self._get_flattened_size(height, width)
        
        self.fc1 = nn.Linear(flattened_size + 1, 4096)  # +1 to include speed feature
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3072)
        self.fc4 = nn.Linear(3072, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc_output = nn.Linear(1024, 3)  # 3 output units (assuming steering, throttle, brake)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def _get_flattened_size(self, height, width):
        # This function calculates the size after the convolutional layers for fully connected layers
        x = torch.zeros(1, 3, height, width)
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv6(self.conv5(x)))
        x = self.conv8(self.conv7(x))
        return x.numel()  # Return the number of elements after flattening

    def forward(self, features):
        # Assuming features has the shape [batch_size, num_features], where num_features includes speed and image data
        speed = features[:, 0].unsqueeze(1)  # Extract speed (shape: [batch_size, 1])
        image_data = features[:, 1:].reshape(-1, 3, 160, 640)  # Reshape image data into the proper dimensions

        # Forward pass through convolutional layers
        x = F.relu(self.conv1(image_data))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Concatenate speed to the flattened conv output
        x = torch.cat((speed, x), dim=1)
        
        # Forward pass through fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        
        # Output layer
        predictions = self.fc_output(x)
        
        return predictions
