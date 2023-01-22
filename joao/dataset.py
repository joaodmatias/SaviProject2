

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

        class_exist = False
        for name in class_names:
            if name == class_name:
                label = class_names.index(name)
                class_exist = True

        if class_exist == False:
            class_names.append(class_name)
            label = class_names.index(class_name)
                
        return label, class_names
        