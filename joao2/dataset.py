

import torch
import numpy as np
from colorama import Fore, Style
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)

        

        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    
    def __getitem__(self, index):  # return a specific element x,y given the index, of the dataset

        # Load the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t
    def __len__(self):  # return the length of the dataset
        return self.num_images

    