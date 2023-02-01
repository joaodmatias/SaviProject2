#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import math
import numpy as np
import os
import cv2
from more_itertools import locate

import math
from copy import deepcopy
import open3d as o3d
import cv2
import numpy as np
import os
from gtts import gTTS
import pygame
from more_itertools import locate



class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud


    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.05, ransac_n=5, num_iterations=50):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud


    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' points and ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text
    
class FindTable():
    def __init__(self, planes):
        self.planes = planes
    def find(self):
        planes = self.planes
        dists = []

        dist_to_0_1 = planes[0].d
        dist_to_0_2 = planes[1].d

        dists.append(dist_to_0_1)
        dists.append(dist_to_0_2)

        dist_between_planes = abs(dist_to_0_1) - abs(dist_to_0_2)
       
        if abs(dist_between_planes) > 0.25:
            if dist_between_planes > 0:
                self.table = planes[1]
            else:
                self.table = planes[0]
        else:
            self.table = planes[0]

        return self.table
    
    def cutTable(self):
        self.pc = []
        pc = self.table.inlier_cloud
        pc = pc.voxel_down_sample(voxel_size=0.005) 
        
        print('PC === ' + str(pc))
        cluster_idxs = list(pc.cluster_dbscan(eps=0.09, min_points=1000, print_progress=True))
        object_idxs = list(set(cluster_idxs))
        object_idxs.remove(-1)

        table_size = 0
        sizes = []
        for object_idx in object_idxs:
            object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
            object_points = pc.select_by_index(object_point_idxs)

            size = len(object_points.points)
            
            if size > table_size:
                self.pc = object_points

            sizes.append(size)
            table_size = max(sizes)

        return self.pc
    
    def transform(self, r, p, y, tx, ty, tz): 

        # Convert from rad to deg
        r = math.pi * r / 180.0
        p = math.pi * p / 180.0
        y = math.pi * y / 180.0

        # First rotate
        rotation = self.pc.get_rotation_matrix_from_xyz((r,p,y))
        self.pc.rotate(rotation, center=(0, 0, 0))

        # Then translate
        self.pc = self.pc.translate((tx,ty,tz))

class PointCloudProcessing():
    def __init__(self):
        pass

    def loadPointCloud(self, filename):

        os.system('pcl_ply2pcd ' + filename + ' pcd_point_cloud.pcd')
        self.pcd = o3d.io.read_point_cloud('pcd_point_cloud.pcd')

        print("Load a point cloud from " + filename)
        self.original = deepcopy(self.pcd) # make a vbackup of the original point cloud
        
        return self.pcd

    def preProcess(self,voxel_size=0.02):
        # Downsampling using voxel grid filter
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size) 
        print('Downsampling reduced point cloud from  ' + str(len(self.original.points)) + ' to ' + str(len(self.pcd.points))+  ' points')

        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        self.pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        
    def transform(self, r, p, y, tx, ty, tz): 

        # Convert from rad to deg
        r = math.pi * r / 180.0
        p = math.pi * p / 180.0
        y = math.pi * y / 180.0

        # First rotate
        rotation = self.pcd.get_rotation_matrix_from_xyz((r,p,y))
        self.pcd.rotate(rotation, center=(0, 0, 0))

        # Then translate
        self.pcd = self.pcd.translate((tx,ty,tz))

    def crop(self, min_x, min_y, min_z, max_x, max_y, max_z):

        #First create a point cloud with the vertices of the desired bounding box
        np_points = np.ndarray((8,3),dtype=float)

        np_points[0,:] = [min_x, min_y, min_z]
        np_points[1,:] = [max_x, min_y, min_z]
        np_points[2,:] = [max_x, max_y, min_z]
        np_points[3,:] = [min_x, max_y, min_z]

        np_points[4,:] = [min_x, min_y, max_z]
        np_points[5,:] = [max_x, min_y, max_z]
        np_points[6,:] = [max_x, max_y, max_z]
        np_points[7,:] = [min_x, max_y, max_z]

        # From numpy to Open3D
        bbox_points = o3d.utility.Vector3dVector(np_points) 

        # Create AABB from points
        self.bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points)
        self.bbox.color = (1, 0, 0)

        self.pcd = self.pcd.crop(self.bbox)


class ObjectProperties():
    def __init__(self, object):
        self.idx = object['idx']
        self.center = object['center']

        self.point_cloud = object['points']
        pc_points = self.point_cloud.points
        self.points = np.asarray(pc_points)

    def getSize(self):
        self.point_cloud.translate(-self.center)
        pc_points_centered = self.point_cloud.points
        points = np.asarray(pc_points_centered)
        
        max_dist_from_center = 0
        max_z = -1000
        min_z = 1000
        for point in points:
            dist_from_center = math.sqrt(point[0]**2 + point[1]**2)

            if dist_from_center >= max_dist_from_center:
                max_dist_from_center = dist_from_center

            z = point[2]
            if z >= max_z:
                max_z = z
            elif z <= min_z:
                min_z = z
        
        width = max_dist_from_center*2
        height = abs(max_z - min_z)

        self.point_cloud.translate(self.center)

        return (width, height)

    def getColor(self, idx):  
        idx = idx + 1
        image_name = 'image' + str(idx) + '.png'

        # Creating o3d windows with only one object to then process in OpenCV
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.point_cloud)
        vis.get_view_control().rotate(0, np.pi / 4) # rotate around y-axis
        vis.get_view_control().set_zoom(3.0) #set the zoom level
        vis.run()  # user changes the view and press "q" to terminatem)
        vis.capture_screen_image(image_name)
        vis.destroy_window()

        # OpenCV processing
        img = cv2.imread(image_name)
        
        # print(str(img))
        colored_pixels = []

        b = 0
        g = 0
        r = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]

                if pixel[0] < 250 or pixel[1] < 250 or pixel[2] < 250:
                    colored_pixels.append(pixel)
                    b = b + pixel[0]
                    g = g + pixel[1]
                    r = r + pixel[2]

        b = b/len(colored_pixels)
        g = g/len(colored_pixels)
        r = r/len(colored_pixels)

        return (r,g,b)
    
    
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
        text_final = "We are looking ate the scene "+str(cenario)+" we have "+ str(len(lista_audio))+ " objects processed in the scene"+str(cleaned_items)

        tts = gTTS(text_final, lang='en')
        tts.save("narracao.mp3")

        pygame.init()
        pygame.mixer.music.load("narracao.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)    
        
        
        
# --------------------------------------------------------------
# Create class for image processing 
# --------------------------------------------------------------
class ImageProcessing():
    def __init__(self):
        pass

    # --------------------------------------------------------------
    # 3D to pixel 
    # --------------------------------------------------------------
    def loadPointCloud(self, points, show, scenario):
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
        img = cv2.imread("Jota/rgb_data/"+scenario+".png")
        # obter altura, largura e número de canais da imagem
        height, width, _ = img.shape
        # imprimir resolução da imagem
        print("Resolução da imagem: " + str(width) + " x " + str(height))
        #settings
        raio = 1
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
            a = y-60
            b = y+60
            c = x-60
            d = x+60
            # if a < 0:
            #     a = 0
            # if c < 0:
            #     c = 0
            # if b > 420:
            #     a = 420
            # if d < 640:
            #     d = 640
            print(str(a)+" "+str(b)+" "+str(c)+" "+str(d))
            crop_img = img[a:b, c:d]
        
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
        if show:
            # Exibir imagens na janela
            cv2.imshow("Janela", result)

            # Esperar até que uma tecla seja pressionada
            cv2.waitKey(0)

            # Destruir a janela
            cv2.destroyAllWindows()
        
        return point