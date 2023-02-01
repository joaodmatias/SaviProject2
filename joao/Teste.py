#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: SAVI 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model import Model

# Load the model from the .pkl file
model = torch.load("model.pkl")

# Set the model in evaluation mode


# Load the input image and perform necessary pre-processing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_image = Image.open("/home/joao/Documents/SAVI/SaviProject2/Jota/images_algarvio/bowl.png")
input_image = transform(input_image)

# Pass the input image through the model
output = Model(input_image)

with torch.no_grad():
    output = Model(input_image)

# Get the predictions
_, predictions = output.max(dim=1)

# Get the label associated with the image
label = predictions.item()

print(f"The label associated with the input image is: {label}")





