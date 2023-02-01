
# SAVI - Where's my Coffee Mug

The "SAVI project - Where's my coffee mug?" implements an advanced perception system that processes information collected from 3D sensors and conventional cameras. The goal is to extract objects from a generated point cloud and using them to train a neural network classifier. This classifier will then be able to tell what the object is.

<p align="center">
  <img src="https://www.hipersuper.pt/wp-content/uploads/2012/12/Universidade-de-Aveiro.jpg">
</p>

Index
=================

  * [Description](#description)
  * [The Project](#the-project)
      * [Requirements](#requirements)
      * [Dataset](#dataset)
      * [Usage](#usage)
      * [Demo](#demo)
      * [Functionalities/Improvements](#functionalitiesimprovements)
  * [Authors](#authors)
  * [Reference](#reference)


# Description
The second assignment of the SAVI ([Advanced Industrial Vision Systems](https://www.ua.pt/pt/uc/14722)) a curricular unit given at the [university of aveiro](https://www.ua.pt/) in the [Master's degree in mechanical engineering](https://www.ua.pt/pt/curso/488) the project aimed to teach the basics of 3D point cloud understanding and processing, as well as the use of classifiers and integration as a system. The main objective was to recognize objects identified in the point cloud using the "Washington RGB-D Dataset".

![image](https://user-images.githubusercontent.com/92520749/215944005-0af835c8-5634-4e37-bc28-ef263991ea8d.png)

# The Project
This project uses Open3D for point cloud processing of a dataset, OpenCV for image processing and feature extraction and PyTorch for deep neural network training of a classifier that will be able to recognize objects.

## Requirements
It is necessary to install the following softwares before any use:
* Open3D
* OpenCV
* PyTorch
* Pickle
* Matplotlib
* GTTS


Also network connection is required.

## Dataset
For the point cloud and image generation this program uses the [Washington RGB-D Dataset](https://rgbd-dataset.cs.washington.edu/).


## Usage

You can use the following command to download the program:
```
git clone https://github.com/joaodmatias/SaviProject2.git
```
To run the program you can start by moving to the directory where you cloned the repository. Once in there you can use:
```
./main.py -h
```
to get some help on options to run, including to add a path to run different scenarios.
You can then use:
```
./main.py -p DATASET_PATH
```
while replacing "DATASET_PATH" with the path to the scenario you want to run.
If no scenario is chosen, there is a preset scenario that will run.





## Demo

- **Objective 1** - Training a classifier in deep learning [Video](https://www.youtube.com/watch?v=6eeXVDOA_Mk&ab_channel=fratymusic)

- **Objective 2** - 3D pre-processing [Video](https://www.youtube.com/watch?v=6eeXVDOA_Mk&ab_channel=fratymusic)

- **Objetivo 3** - Classificação de objetos na cena [Video](https://www.youtube.com/watch?v=6eeXVDOA_Mk&ab_channel=fratymusic)

- **Final Work** - System in its entirety [Video](https://www.youtube.com/watch?v=6eeXVDOA_Mk&ab_channel=fratymusic) 
## Functionalities/Improvements

- [x] Different objects classification
- [x] 3D dataset processing
- [x] Extracting information from the point cloud such as:
    - [x] color
    - [x] dimensions
    - [ ] volume
    - [ ] orientation
    
 The color information will appear on the terminal where you run the program, as an approximation to the CSS21 list of colors as well as the actual RGB value. <br/>
 The dimensions will appear as a tuple such as (width, height) in meters.

Here we can see the extraction of images of objects used to train the classifier: <br/>
![image](https://user-images.githubusercontent.com/92520749/215945372-cfd947f6-9fe8-4e6c-9573-e4fdfc92bb5e.png)


# Authors

- [@jotadateta](https://github.com/jotadateta) - joaopedrotomas@ua.pt 93366
- [@joaodmatias](https://github.com/joaodmatias) - joaodamatias@ua.pt 93098
- [@joaodrc](https://github.com/joaodrc) - joaodanielc@ua.pt 93439


# Reference

 - [Proposed work](https://github.com/miguelriemoliveira/savi_22-23/tree/main/Trabalho2)

