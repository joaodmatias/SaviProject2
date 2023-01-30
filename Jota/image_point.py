
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
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from point_cloud_processing import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate


class ImageProcessing():
    def __init__(self):
        pass

    def loadPointCloud(self, points):
        #print("centros: "+ str(points))
        points_group = []
        for point in points :
            x,y,z = point[0],point[1],point[2]
            #x,y,z = 1,1,1
            points_3d = np.array([[x, y, z]], dtype=np.float32)
            
            #Matriz intrinsica
            fx = 570.3
            fy = 570.3
            cx = 640 / 2
            cy = 480 / 2

            intrinsic_matrix = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]], dtype=np.float32)
            
            # cálculo das coordenadas dos pontos em 2D
            #points_2d, _ = cv2.projectPoints(points_3d, np.zeros((3, 1)), np.zeros((3, 1)), intrinsic_matrix, np.zeros((1, 4)))
            points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), intrinsic_matrix, None, None)
            points_2d = np.round(points_2d).astype(int)
            
            print("pontos: "+str(points_2d))
            #imagePoints,_ = cv2.projectPoints(new_objs_center, rvec, tvec, intrinsic_matrix, distCoeffs)
            
            points_group.append(points_2d)
            
        points_group = [point[0][0] for point in points_group]
        print("points total" + str(points_group))
        
        # carregar imagem
        img = cv2.imread("../Jota/rgb_data/00000-color.png")
        # obter altura, largura e número de canais da imagem
        height, width, _ = img.shape
        # imprimir resolução da imagem
        print("Resolução da imagem: " + str(width) + " x " + str(height))
        #settings
        raio = 5
        cor = (0,0,255)
        # desenhar círculos preenchidos nos pontos
        for point in points_group:
            x, y = point
            cv2.circle(img, (x, y), raio, cor, -1)

        # mostrar imagem
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()