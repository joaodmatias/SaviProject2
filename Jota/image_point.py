
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
import os
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from point_cloud_processing import PointCloudProcessing
from matplotlib import cm
from more_itertools import locate


# --------------------------------------------------------------
# Create class for image processing 
# --------------------------------------------------------------
class ImageProcessing():
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # 3D to pixel 
    # --------------------------------------------------------------
    def loadPointCloud(self, points):
        # Inputs are the center points processed in the main file
        
        #print("centros: "+ str(points))
        points_group = []
        
        # --------------------------------------------------------------
        # Processing each point we gave as input 
        # --------------------------------------------------------------
        for point in points :
            x,y,z = point[0],point[1],point[2]
            #x,y,z = 1,1,1 # just for debuging
            
            points_3d = np.array([[x, y, z]], dtype=np.float32)
            
            #Matriz intrinsica
            fx = 570.3  #Value we get from André, thanks for that!
            fy = 570.3
            cx = 320 #Resolution image divided by 2
            cy = 240

            intrinsic_matrix = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])            
            
            
            # distcoof
            distCoeffs = np.array([0, 0, 0, 0])
            rvec = np.identity(3)
            tvec = np.zeros(3)
            # --------------------------------------------------------------
            # Using projectpoints function to transform the 3D to pixel 
            # --------------------------------------------------------------
            points_2d,_ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, None)
            print("pontos: "+str(points_2d))
            
            #Associate the points to a list
            points_group.append(points_2d)
           
            
        points_group = [point[0][0] for point in points_group]
        print("points total" + str(points_group))
        
        # --------------------------------------------------------------
        # Now lets processed the image with the points we get 
        # --------------------------------------------------------------
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
        images = []
        
        # Image directory
        directory = "/home/jota/Documents/SAVI/Savi_trabalho_2/SaviProject2/Jota/images_algarvio"
            
        for idx, point in enumerate(points_group):          
            x, y = int(point[0]), int(point[1])
            print("point x: "+ str(int(x))+ " y: "+ str(int(y)))
            cv2.circle(img, (x, y), raio, cor, -1)
            
            # Definir o retângulo de crop
            a = y-80
            b = y+80
            c = x-80
            d = x+80
            print(str(a)+" "+str(b)+" "+str(c)+" "+str(d))
            crop_img = img[a:b, c:d]
            cv2.imshow("Janela", crop_img)
        
            # Change the current directory 
            # to specified directory 
            os.chdir(directory)
            
            # List files and directories  
            # in 'C:/Users/Rajnish/Desktop/GeeksforGeeks'  
            print("Before saving image:")  
            print(os.listdir(directory))  
            
            # Filename
            filename = "crop_image_"+str(idx)+".jpg"
            
            # Using cv2.imwrite() method
            # Saving the image
            cv2.imwrite(filename, crop_img)
            
            images.append(crop_img)
        
        # Concatenar as imagens horizontalmente
        result = cv2.hconcat(images)

        # Exibir imagens na janela
        cv2.imshow("Janela", result)

        # Esperar até que uma tecla seja pressionada
        cv2.waitKey(0)

        # Destruir a janela
        cv2.destroyAllWindows()
        
        return point