#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import math
import numpy as np
import os
import cv2


class PointCloudProcessing():
    def __init__(self):
        pass

    def loadPointCloud(self, filename):

        os.system('pcl_ply2pcd ' + filename + ' pcd_point_cloud.pcd')
        self.pcd = o3d.io.read_point_cloud('pcd_point_cloud.pcd')

        print("Load a point cloud from " + filename)
        self.original = deepcopy(self.pcd) # make a vbackup of the original point cloud

    def preProcess(self,voxel_size=0.02):
        # Downsampling using voxel grid filter
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size) 
        print('Downsampling reduced point cloud from  ' + str(len(self.original.points)) + ' to ' + str(len(self.pcd.points))+  ' points')

        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        self.pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
        
    def transform(self, r,p,y,tx,ty,tz): 

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


    def findPlane(self, distance_threshold=0.01, ransac_n=3, num_iterations=100):

        print('Segmenting plane from point cloud with ' + str(len(self.pcd.points)) + ' points')
        plane_model, inlier_idxs = self.pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model
        self.inliers = self.pcd.select_by_index(inlier_idxs)
        
        outlier_cloud = self.pcd.select_by_index(inlier_idxs, invert=True)
        return outlier_cloud

class ObjectProperties():
    def __init__(self, object):
        self.idx = object['idx']
        # self.pc_color = object['color']
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