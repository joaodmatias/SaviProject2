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
from point_cloud_processing import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate
from image_point import ImageProcessing

class audioprocessing():
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # 3D to pixel 
    # --------------------------------------------------------------
    def loadaudio(self):
        algarvio = "Vai te foder"
        # Gerar a descrição da cena
        descricao = "O joão conceição " + algarvio + " obrigado."

        # Gerar o áudio sintetizado da descrição
        tts = gTTS(descricao, lang='pt-br')
        tts.save("narracao.mp3")

        # Tocar o áudio
        pygame.init()
        pygame.mixer.music.load("narracao.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
