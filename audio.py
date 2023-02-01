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
from gtts import gTTS
import pygame
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib import cm
from more_itertools import locate
from image_point import ImageProcessing

class audioprocessing():
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # 3D to pixel 
    # --------------------------------------------------------------
    def loadaudio(self, lista_audio, cenario):
        
        print(lista_audio)
        text = "We found a " + " and a ".join(lista_audio) + "."
        cleaned_items = [item.split("_")[0] for item in lista_audio]
        print(cleaned_items)
        pygame.mixer.init()
        
        # Gerar a descrição da cena
        text_final = "We are looking ate the scene "+str(cenario)+" we have "+ str(len(lista_audio))+ " objects processed in the scene"+str(text)

        tts = gTTS(text_final, lang='en')
        tts.save("narracao.mp3")

        pygame.init()
        pygame.mixer.music.load("narracao.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)    
        

        
