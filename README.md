# Self-Driving Car in GTA using PyTorch , crazy right? probably not that crazy

![Self-Driving Car Program](README/README.png)


This is a fully-featured **self-driving car program** built using **PyTorch** and **Convolutional Neural Networks (CNNs)**. The system is capable of controlling a car's movement autonomously by processing input data from the game environment.


## Installation

To set up the environment and install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Use the https://github.com/scripthookvdotnet to allow the logging script to run in order to collect output speed of the vehicle

This project requires the installation of `ScpVBus` if you intend on using the Virtual Controller object. It can be installed by following the below. 
More information can be found at [ScpVBus](https://github.com/nefarius/ScpVBus).

Along with the development of the https://github.com/DreadPirate09/Semantic-Segmentation-For-Road-Detection ,We can use now just the features of the lane marks + the minimap for a more accurate control of the car inside the road lanes.
We have now a script that will prepare the datasets extracting the lane marks using the pretrained u-net model after which we will train our old model.
Here is a sample from the new dataset of images.

![Sample ](README/lane_marks_features.png) 