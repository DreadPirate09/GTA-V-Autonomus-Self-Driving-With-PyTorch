import os
import cv2
import torch
import win32gui, win32api
from torchvision import transforms
from PIL import Image
from gta_v_driver_model import GTAVDriverModel
from u_net_model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pilot import Pilot
import mss
import pygame
import keyboard

#################################################################################

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

u_net_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load('my_checkpoint.pth.tar')
u_net_model.load_state_dict(checkpoint['state_dict'])
u_net_model.eval()
transform_u_net = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0
                ),
            ToTensorV2()
        ]
    )

##################################################################################

model = GTAVDriverModel().to(DEVICE)
model.load_state_dict(torch.load('gtav_driver_model.pth'))
model.eval() 

transform = transforms.Compose([
    transforms.Resize((160, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

##################################################################################

def read_last_speed(log_file_path):

    with open(log_file_path, 'rb') as f:
        f.seek(-2, os.SEEK_END) 
        while f.read(1) != b'\n': 
            f.seek(-2, os.SEEK_CUR) 
        last_line = f.readline().decode().strip() 
    return float(last_line) 


def preprocess_frame(frame):

    pil_image = Image.frombytes('RGB', frame.size, frame.rgb)
    pil_image = pil_image.resize((640, 360), Image.BICUBIC)
    org_img = np.array(pil_image)

    pil_image = pil_image.crop(box=(0, 150, 640, 360))
    img_np = np.array(pil_image)
    transformed = transform_u_net(image=img_np)
    input_tensor = transformed['image'] 
    input_tensor = input_tensor.unsqueeze(0) 
    input_tensor = input_tensor.to(DEVICE)
    preds = torch.sigmoid(u_net_model(input_tensor))
    preds = (preds > 0.5).float()
    mask_np = preds.squeeze().cpu().numpy()
    resized_mask = cv2.resize(mask_np, (640, 210), interpolation=cv2.INTER_NEAREST)
    mask = resized_mask == 1
    neg_mask = mask == 0
    map_crop = Image.fromarray(org_img).crop(box=(0, 305, 140, 360))

    org_img[150:150 + mask.shape[0], :, 0] = np.where(mask, 0, org_img[150:150 + mask.shape[0], :, 0])
    org_img[150:150 + mask.shape[0], :, 1] = np.where(mask, 255, org_img[150:150 + mask.shape[0], :, 1])
    org_img[150:150 + mask.shape[0], :, 2] = np.where(mask, 0, org_img[150:150 + mask.shape[0], :, 2])
    org_img[150:150 + mask.shape[0], :, 0] = np.where(neg_mask, 0, org_img[150:150 + mask.shape[0], :, 0])
    org_img[150:150 + mask.shape[0], :, 1] = np.where(neg_mask, 0, org_img[150:150 + mask.shape[0], :, 1])
    org_img[150:150 + mask.shape[0], :, 2] = np.where(neg_mask, 0, org_img[150:150 + mask.shape[0], :, 2])

    to_save = Image.fromarray(org_img)
    _, map_height = map_crop.size
    to_save_width, to_save_height = to_save.size

    map_paste_coords = (0, to_save_height - map_height)
    to_save.paste(map_crop, map_paste_coords)
    to_save = to_save.crop(box=(0,150, to_save_width, to_save_height))

    transformed_image = transform(to_save)

    return transformed_image

def run_inference(model, frame, speed):

    frame_tensor = preprocess_frame(frame).unsqueeze(0).to(DEVICE)
    speed_tensor = torch.tensor([[speed]], dtype=torch.float32).to(DEVICE)
    model_input = torch.cat((speed_tensor, frame_tensor.flatten(start_dim=1)), dim=1)

    with torch.no_grad():
        predictions = model(model_input)

    steering, throttle = predictions[0].cpu().numpy()
    return steering, throttle

log_speed_path = os.getcwd()+'\\VehicleSpeedLog.txt'

sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}
driver = Pilot()

pause = False
return_was_down=False

while True:

    if pause is False:
        sct_img = sct.grab(mon)
        speed = read_last_speed(log_speed_path)
        steering, throttle = run_inference(model, sct_img, speed)
        driver.sendIt(steering , throttle, 0, speed)

    if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
        if (return_was_down == False):
            if (pause == False):
                pause = True
                driver.resetController()
                driver.pullHandBreak()
                print("Pause")
            else:
                pause = False
                driver.releaseHandBreak()
                
        return_was_down = True
    else:
        return_was_down = False




