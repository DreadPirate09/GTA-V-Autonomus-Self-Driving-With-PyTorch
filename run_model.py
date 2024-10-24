import os
import torch
from torchvision import transforms
from PIL import Image
from gta_v_driver_model import GTAVDriverModel
from pilot import Pilot
import mss
import pygame

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GTAVDriverModel().to(device)
model.load_state_dict(torch.load('gtav_driver_model.pth'))
model.eval()  # Set the model to evaluation mode

# Transformation for the input image (same as used during training)
transform = transforms.Compose([
    transforms.Resize((160, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_last_speed(log_file_path):
    """
    Reads the last line of the VehicleSpeedLog.txt file to get the latest speed.
    """
    with open(log_file_path, 'rb') as f:
        f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
        while f.read(1) != b'\n':  # Until EOL is found...
            f.seek(-2, os.SEEK_CUR)  # ...jump back the read byte plus one more.
        last_line = f.readline().decode().strip()  # Read the last line
    return float(last_line)  # Convert the last line to a float


def preprocess_frame(frame):
    """
    Preprocesses the captured frame to match the input format of the model.
    """
    # Convert to PIL Image
    pil_image = Image.frombytes('RGB', frame.size, frame.rgb)
    
    # Resize, crop, and transform as used in data collection
    pil_image = pil_image.resize((640, 360), Image.BICUBIC)
    pil_image = pil_image.crop(box=(0, 200, 640, 360))
    
    # Apply the same transformation used during training
    transformed_image = transform(pil_image)
    
    # Return as a tensor
    return transformed_image

def run_inference(model, frame, speed):
    """
    Runs inference using the trained model.
    """
    # Flatten the frame into the appropriate shape
    frame_tensor = preprocess_frame(frame).unsqueeze(0).to(device)  # Add batch dimension
    
    # Speed must also be a tensor and concatenated to the image data
    speed_tensor = torch.tensor([[speed]], dtype=torch.float32).to(device)
    
    # Concatenate speed with image data for the model input
    model_input = torch.cat((speed_tensor, frame_tensor.flatten(start_dim=1)), dim=1)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(model_input)
    
    # Return the predictions (steering, throttle, brake)
    steering, throttle, brake = predictions[0].cpu().numpy()
    return steering, throttle, brake

# Example usage:
log_file_path = 'C:\\GitRepo1\\PyTorch-Gta-Self-Drive\\PyTorch-Explore-Models\\VehicleSpeedLog.txt'

# Initialize mss for screen capturing
sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}  # Modify based on your screen size
driver = Pilot()
# Main loop to capture frame, read speed, and run inference
while True:
    # Capture frame from the game (same as data recorder)
    sct_img = sct.grab(mon)

    # Read the last logged speed
    speed = read_last_speed(log_file_path)
    
    # Run inference using the frame and speed
    steering, throttle, brake = run_inference(model, sct_img, speed)
    driver.sendIt(steering, throttle, brake)
    
    # Print the model's predictions
    print(f"Steering: {steering:.4f}, Throttle: {throttle:.4f}, Brake: {brake:.4f}")
    
    # Add code to send these controls to the game if necessary

