#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: SAVI 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
# -----------------------------------------------------------------

import argparse
import glob
import pickle
import random
from copy import deepcopy
from statistics import mean
import os
from PIL import Image
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm
from model import Model
from colorama import Fore, Style
from torchvision import transforms
import torch.nn.functional as F
from dataset import Dataset
from classification_visualizer import ClassificationVisualizer



def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    # Define hyper parameters
    resume_training = True
    model_path = 'model.pkl'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = Model() # Instantiate model
    class_names = []
    class_names2 = []

    test_visualizer = ClassificationVisualizer('Test Images')

    transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])
 
    # -----------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------

    dataset_path = '/home/joao/Documents/SAVI/rgbd-dataset'
    Image_jota_path = '/home/joao/Documents/SAVI/SaviProject2/Jota/imagens_class'

    name_filenames = glob.glob(dataset_path + '/*/')

    for paths in name_filenames:
            parts = paths.split('/')
            part = parts[6]
            class_names.append(part)
    # print(class_names)

    image_filenames = glob.glob(Image_jota_path + '/*')
    
    # Create the dataset

    dataset_test = Dataset(image_filenames)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=6)

    # dataset_test = Dataset(image_filenames, class_names)
    #loader_test = torch.utils.data.DataLoader(dataset=image_t, batch_size=1000, shuffle=True)

    tensor_to_pil_image = transforms.ToPILImage()


    # Resume training

    checkpoint = torch.load(model_path, map_location= device)
    model.load_state_dict(checkpoint['model_state_dict'])


        # model.train()

    # -----------

    model.to(device) # move the model variable to the gpu if one exists

    
    # Run test in batches ---------------------------------------
    # TODO dropout
    test_losses = []
    #for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

    for image_t in loader_test:
        image_t = image_t.to(device)


        # Apply the network to get the predicted ys
        label_t_predicted = model.forward(image_t)
        
        test_visualizer.draw(image_t, label_t_predicted, class_names)
        input('Enter para continuar')
            #-----------------------------------------------------------------------
            # Vizualizer

 
    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------
    

if __name__ == "__main__":
    main()