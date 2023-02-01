#!/usr/bin/env python3

import copy
import csv
import math
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import webcolors
from matplotlib import cm
from more_itertools import locate
from image_point import ImageProcessing
from audio import  audioprocessing

from Classes import *

def main():
    lista_audio = ['bowl','banana']
    number = 3
    
    audio = audioprocessing()
    audio_final = audio.loadaudio(lista_audio, number)
    
    
if __name__ == "__main__":
    main()