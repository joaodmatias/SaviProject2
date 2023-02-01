

import torch
import numpy as np
from colorama import Fore, Style
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames, class_names):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)

        self.labels = []
        for image_filename in self.image_filenames:
            self.labels.append(self.getClassFromFilename(image_filename, class_names))

        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    
    def __getitem__(self, index):  # return a specific element x,y given the index, of the dataset

        # Load the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t, self.labels[index]

    def __len__(self):  # return the length of the dataset
        return self.num_images

    def getClassFromFilename(self, filename, class_names):

        parts = filename.split('/')
        part = parts[6]

        class_name = part

        # print('filename ' + filename + ' is a ' + Fore.RED + class_name + Style.RESET_ALL)
        Class_name_exists = False
        for name in class_names:
            if name == class_name and Class_name_exists == True:
                raise ValueError('Has more than one class')
            elif name == class_name:
                Class_name_exists = True
                label = class_names.index(name)
                # if label1 < 2:
                #     label = label1
                # else:
                #     label = 5

        if Class_name_exists == False:
            raise ValueError('unknown class')
        
                
        return label
        