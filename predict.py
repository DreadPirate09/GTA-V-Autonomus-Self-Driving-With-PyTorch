import torch
import torchvision.transforms as transforms
from gta_v_driver_model import GTAVDriverModel
from PIL import Image


# Instantiate the model and load saved weights
model = GTAVDriverModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

# Define image preprocessing transformations (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((160, 640)),  # Resize to match input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as in training
])

def preprocess_image(image_path):
    """ Preprocess the image from the path """
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    image = transform(image)  # Apply transformations
    return image

def predict(image_path, speed):
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Add a batch dimension (1, channels, height, width)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Convert speed to a tensor and add a batch dimension
    speed_tensor = torch.tensor([[speed]], dtype=torch.float32)
    
    # Concatenate speed and image tensor as input to the model
    features = torch.cat((speed_tensor, image_tensor.view(1, -1)), dim=1)

    # Run inference
    with torch.no_grad():  # Disable gradient calculation
        output = model(features)
    
    # The output is [steering, throttle, brake]
    steering, throttle, brake = output[0].numpy()
    
    return steering, throttle, brake

# Example usage:
image_path = 'path_to_image.jpg'
speed = 30.0  # Example speed
steering, throttle, brake = predict(image_path, speed)

print(f"Predicted Steering: {steering}")
print(f"Predicted Throttle: {throttle}")
print(f"Predicted Brake: {brake}")