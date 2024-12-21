import os
import cv2
from PIL import Image
import torch
from u_net_model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np



BATCH = 10
DATA_FOLDER = "data/"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load('my_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
transform = A.Compose(
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


files = [x for x in os.listdir('data') if ".bmp" in x]

for f in files:
	img = Image.open(DATA_FOLDER+f)

	###### prepare and save data here
	# img.save(DATA_FOLDER+f.replace("gmi","img"))
	img = img.resize((640, 360), Image.BICUBIC)
	org_img = img
	org_img = np.array(org_img)
	img = img.crop(box=(0, 150, 640, 360))
	img_np = np.array(img)
	transformed = transform(image=img_np)
	input_tensor = transformed['image'] 
	input_tensor = input_tensor.unsqueeze(0) 
	input_tensor = input_tensor.to(DEVICE)
	preds = torch.sigmoid(model(input_tensor))
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
	######
	map_paste_coords = (0, to_save_height - map_height)
	to_save.paste(map_crop, map_paste_coords)
	to_save = to_save.crop(box=(0,150, to_save_width, to_save_height))
	os.remove(DATA_FOLDER+f)
	to_save.save(DATA_FOLDER+f)

print(files)
