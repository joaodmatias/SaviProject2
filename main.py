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
#from image_point import ImageProcessing
#from audio import  audioprocessing

from Classes import *


view_matias = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.5, 0.5, 0.5 ],
			"boundingbox_min" : [ -0.5, -0.5, -0.029999999999999999 ],
			"field_of_view" : 60.0,
			"front" : [ -0.55870796654905075, -0.62136378364421685, 0.54931999462059256 ],
			"lookat" : [ 0.0, 0.0, 0.23499999999999999 ],
			"up" : [ 0.36279278885743554, 0.41250367094854157, 0.83559686081687845 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

view_jota = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.611335277557373, 1.2635015249252319, 3.83980393409729 ],
			"boundingbox_min" : [ -2.5246021747589111, -1.5300980806350708, -1.4928504228591919 ],
			"field_of_view" : 60.0,
			"front" : [ 0.015753200428247988, -0.064656272400542641, -0.99778324455541667 ],
			"lookat" : [ -0.0021667868580298368, 0.017179249546857372, 0.42906548023665808 ],
			"up" : [ -0.087240296429267908, -0.99419028473249504, 0.063046081737518522 ],
			"zoom" : 0.02
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

############################3333
##Function to vizualize the ICP comparation##
def draw_registration_result(source, target, transformation, bbox_to_draw_object, bbox_to_draw_object_target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, bbox_to_draw_object, bbox_to_draw_object_target],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
#############################################

##Load 3D scenario cloud to screenshot it###
def load_view_point(pcd, filename, id):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    ctr.set_zoom(0.15)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image("image_extra/image_"+str(id)+".jpg")        
    
    vis.destroy_window()
############################################


##Pre-set the view of the 3D cloud object to see in real ambient##
def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", help="Activate camera ref", action="store_true")
parser.add_argument("-e", "--extra", help="Activate extra function 1", action="store_true")
parser.add_argument("-t", "--comparation", help="Activate comparation mode", action="store_true")
parser.add_argument("-f", "--cropped", help="Show the final cropped images", action="store_true")
parser.add_argument("-p", "--dataset_path", help="Select the path of the desire point cloud", default="Matias/Scenes/pc/03.ply", type=str)


args = parser.parse_args()
# ------------------------------------------
# MInstructions
#   c - activates camera in the ref perspective only for visualization
#   e - Activates the extra 1, it shows a window where after pressing "q" you save yout view point and save the 
#       image(was expecting time for a more advanced study of that in the classificator)
#   t - Activates the comparation mode , where you can visualize the diferent interations of each dataset in order
#       to found the model using the ICP 
#   f - Show de images captured after transforming the 3D camera for the 2D camera processing to the cropped of each object
#   
# ------------------------------------------

# ------------------------------------------
# Main function 
# ------------------------------------------


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    pc = PointCloudProcessing()
    
    # ----------------------------------------------
    # Extract the file name pcd for image processing
    # ----------------------------------------------
    file_name = args.dataset_path.split('/')[-1]
    number = file_name.split('.')[0]
    
    # ----------------------------------------------
    # Load PC file from respective scenario
    # ----------------------------------------------
    print("Load a ply point cloud, print it, and render it")
    scenario_name = str(number) 
    point_cloud = pc.loadPointCloud(args.dataset_path)
  
    # ------------------------------------------
    # Execution
    # ------------------------------------------    

    planes = []
    planes_pc = deepcopy(point_cloud)
    colormap = cm.Pastel1(list(range(0,2)))

    while True:
        plane = PlaneDetection(planes_pc)
        planes_pc = plane.segment()

        idx_color = len(planes)
        color = colormap[idx_color, 0:3]

        plane.colorizeInliers(r=color[0], g=color[1], b=color[2])
        planes.append(plane)

        if len(planes) >= 2: # stop detection planes
            break

    table_inst = FindTable(planes)
    
    table_inst.find()

    table = table_inst.cutTable()

    # Axis Alignment with table center
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    tx, ty, tz = table.get_center()

    pc.transform(0, 0, 0, -tx, -ty, -tz)
    pc.transform(-108, 0, 0, 0, 0, 0)
    pc.transform(0, 0, -37, 0, 0, 0)
    
    
    
    if scenario_name == '01' or scenario_name == '02'or scenario_name == '06':
        pc.transform(-5, 0, 0, 0, 0, 0)
    elif scenario_name == '04':
        pc.transform(-5, 0, 0, 0, 0, 0)
        pc.transform(0, 5, 0, 0, 0, 0)
    elif scenario_name == '05':
        pc.transform(5, 0, 0, 0, 0, 0)
        pc.transform(0, 5, 0, 0, 0, 0)
    elif scenario_name == '07':
        pc.transform(0, 5, 0, 0, 0, 0)
    elif scenario_name == '08':
        pc.transform(-5, 0, 0, 0, 0, 0)
        pc.transform(0, 5, 0, 0, 0, 0)
    elif scenario_name == '09':
        pc.transform(-7, 0, 0, 0, 0, 0)
        pc.transform(0, 10, 0, 0, 0, 0)
    elif scenario_name == '10':
        pc.transform(-10, 0, 0, 0, 0, 0)
        pc.transform(0, 10, 0, 0, 0, 0)
    elif scenario_name == '11':
        pc.transform(-7, 0, 0, 0, 0, 0)
        pc.transform(0, 10, 0, 0, 0, 0)
    elif scenario_name == '12':
        pc.transform(-7, 0, 0, 0, 0, 0)
        pc.transform(0, 7, 0, 0, 0, 0)
    elif scenario_name == '13':
        pc.transform(0, 7, 0, 0, 0, 0)
        pc.transform(-15, 0, 0, 0, 0, 0)
    elif scenario_name == '14':
        pc.transform(-5, 0, 0, 0, 0, 0)
        pc.transform(0, 4, 0, 0, 0, 0)
        
    pc.crop(-0.5, -0.5, 0.025, 0.45, 0.5, 0.4)

    outliers = pc.pcd  
    print(outliers)
    
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=10, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)


    number_of_objects = len(object_idxs)

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)
        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx + 1)
        d['points'] = object_points 
        d['center'] = d['points'].get_center()
        

        pc_to_convert = d["points"]
        pc_points = pc_to_convert.points
        points = np.asarray(pc_points)
        
        if points.size > 700:
            objects.append(d) # add the dict of this object to the list
        else:
            continue

    # # ------------------------------------------
    # # Visualization
    # # ------------------------------------------
    # # Create a list of entities to draw
    # entities = [table]
    entities = []
    
    entities.append(frame)
    
    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pc.bbox)
    entities.append(bbox_to_draw)
    dimensions = []
    # Draw objects
    color = []
    for object_idx, object in enumerate(objects):
        entities.append(object['points'])

        properties = ObjectProperties(object)
        size = properties.getSize()
        print("This object's SIZE is " + str(size))
        dimensions.append(size)

        color_rgb = properties.getColor(object_idx)
        
        min_colours = {}
        for key, name in webcolors.CSS21_HEX_TO_NAMES.items():#CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - color_rgb[0]) ** 2
            gd = (g_c - color_rgb[1]) ** 2
            bd = (b_c - color_rgb[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        closest_color = min_colours[min(min_colours.keys())]

        try:
            actual_name = webcolors.rgb_to_name(color_rgb)
            closest_name = actual_name
        except ValueError:
            closest_name = closest_color
            actual_name = None

        print("This object's approximate COLOR is " + str(closest_name) + ' with ' + 
              str(color_rgb) + ' RGB value')
        color.append(closest_name)
        # Get the aligned bounding box of the point cloud
        bbox_to_draw_object_processed = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
        entities.append(bbox_to_draw_object_processed)

    entities.append(pc.pcd)
    
    # Create an instance of the class
    pct = PointCloudProcessing()

    # Set the point cloud as an attribute of the class instance
    pct.pcd = point_cloud
    
    # Call the transform function on the class instance, passing in the transformation parameters
    # pct.transform(0, 0, 0, -tx, -ty, -tz)
    # pct.transform(-108, 0, 0, 0, 0, 0)
    # pct.transform(0, 0, -37, 0, 0, 0)
    
    # Associate the point cloud to the objects for better view         
    entities.append(pct.pcd)
    
    # Associate the point cloud to the objects for better view         
    

    o3d.visualization.draw_geometries(entities,
                                    zoom=view_matias['trajectory'][0]['zoom'],
                                    front=view_matias['trajectory'][0]['front'],
                                    lookat=view_matias['trajectory'][0]['lookat'],
                                    up=view_matias['trajectory'][0]['up'])
    
    # ------------------------------------------
    # Termination
    # ------------------------------------------
    
    to_show = []
    centers = []
    
    # ------------------------------------------
    # Move the scenario point cloud   
    # ------------------------------------------
    # Create an instance of the class
    pcp = PointCloudProcessing()

    # Set the point cloud as an attribute of the class instance
    point_cloud_after = deepcopy(point_cloud)
    pcp.pcd = point_cloud_after
    
    # Call the transform function on the class instance, passing in the transformation parameters
    pcp.transform(0,0,37,0,0,0)
    pcp.transform(108,0,0,0,0,0)
    pcp.transform(0,0,0,tx,ty,tz)
    
    
    # ------------------------------------------
    # Read the objects point clouds  
    # ------------------------------------------
    # Ex6 - ICP
    
    path = 'Data_objects'
    files = [f for f in os.listdir(path) if f.endswith('.pcd')]

    # dictionary all the desired objects for processing
    list_pcd = {}
    for i, file in enumerate(files):
        variable_name = os.path.splitext(file)[0]
        point_cloud = o3d.io.read_point_cloud(os.path.join(path, file))
        list_pcd[variable_name] = {'point_cloud': point_cloud, 'indexed': i} 
        
    
    list_pcd_model = []
    for variable_name, info in list_pcd.items():
        list_pcd_model.append(info["point_cloud"])
        
    
    
    for object_idx, object in enumerate(objects):
        object['rmse'] = 10
        object['indexed'] = 100
        min_error = 0.03
        for model_idx, models_object in enumerate(list_pcd_model): 
            #print("Apply point-to-point ICP to object " + str(object['idx']) )

            trans_init = np.asarray([[1, 0, 0, 0],
                                    [0,1,0,0],
                                    [0,0,1,0], 
                                    [0.0, 0.0, 0.0, 1.0]])
            reg_p2p = o3d.pipelines.registration.registration_icp(object['points'], models_object, 1.0, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            #print(str(reg_p2p.inlier_rmse) + "A quando comparado com: " + str(model_idx) )
            #print(reg_p2p)
            # -----------------------------------------------------
            # Start processing each object and respectiv properties
            # -----------------------------------------------------
            ##Bounding box to see better the comparation###
            bbox_to_draw_object = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
            bbox_to_draw_object.color = (1, 0, 0)
            bbox_to_draw_object_target = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(models_object)
            bbox_to_draw_object_target.color = (0, 1, 0)
            
            ##Get some information about the bound boxes###
            Volume_source = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object)
            Volume_target = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object_target)
            Dimensions_source = o3d.geometry.AxisAlignedBoundingBox.get_extent(bbox_to_draw_object)
            Dimensions_target = o3d.geometry.AxisAlignedBoundingBox.get_extent(bbox_to_draw_object_target)
            
            # Some Debuging
            #print("Volume da Source: " + str(Volume_source))
            #print("Volume do Target " + str(Volume_target))   
            #print("dimensões do Source: " + str(Dimensions_source))
            #print("dimensões do Target: " + str(Dimensions_target))

            # ------------------------------------------
            # Doing Some match for better analysis
            # ------------------------------------------
            volume_compare = abs(Volume_source - Volume_target)  
            #print("Volume de comparação " + str(volume_compare))   
            
            x_distance = abs(Dimensions_source[0]- Dimensions_target[0])
            y_distance = abs(Dimensions_source[1]- Dimensions_target[1])
            z_distance = abs(Dimensions_source[2]- Dimensions_target[2])
            
            #print("distancia em x: " + str(x_distance))
            #print("distancia em y: " + str(y_distance))
            #print("distancia em z: " + str(z_distance))
            #if z_distance < 0.01 :  
            
            # ------------------------------------------
            # Start associating each object 
            # ------------------------------------------
            
            if  volume_compare < 0.006 :        
                if reg_p2p.inlier_rmse < min_error and reg_p2p.inlier_rmse != 0:
                    if object['rmse'] > reg_p2p.inlier_rmse:
                        object['rmse'] = reg_p2p.inlier_rmse
                        object['indexed'] = model_idx
                        object["fitness"] = reg_p2p.fitness
                        print(object)
                        if args.comparation:
                            draw_registration_result( object['points'],models_object, reg_p2p.transformation, bbox_to_draw_object, bbox_to_draw_object_target)

    
    
    # --------------------------------------------------------------
    # Restrasnform for the inital ref - jota
    # --------------------------------------------------------------
    for object_idx, object in enumerate(objects):

        point_cloud_demo = deepcopy(object['points'])    
        pcp.pcd = point_cloud_demo
        
        
        pcp.transform(0,0,37,0,0,0)
        pcp.transform(108,0,0,0,0,0)
        pcp.transform(0,0,0,tx,ty,tz)
        to_show.append(pcp.pcd)
        
        # Get the orieted bouding box
        bbox_to_draw_object_test = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(pcp.pcd)
        
        # Get the center of the each object
        bbox_to_draw_object_center = o3d.geometry.AxisAlignedBoundingBox.get_center(pcp.pcd)
        #print("centro: " + str(bbox_to_draw_object_center))
        
        # Put each center for visualization
        centers.append(bbox_to_draw_object_center)
        
        # Create a sphere in each center for better visualization
        sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.04)
        sphere.paint_uniform_color([1, 0, 0]) # muda a cor para vermelho
        sphere = sphere.translate(bbox_to_draw_object_center)

        to_show.append(sphere)

        to_show.append(bbox_to_draw_object_test)
        
        
        
    # Associate the inital objects to the original PC
     # mesh_box = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    to_show.append(point_cloud_after)
    to_show.append(frame)
    
    
    # ----------------------------------------------------
    # Backup code in case we cant do the processing image 
    # EXTRA - activate in case of use the argprase extra
    # ----------------------------------------------------
    if args.extra:    
        for idx, object in enumerate(objects):
            save_view_point(object['points'], "viewpoint.json")
            load_view_point(point_cloud, "viewpoint.json", idx) 
            
    # ------------------------------------------
    # Visualize the point cloud  
    # ------------------------------------------
    if args.camera:
        o3d.visualization.draw_geometries(to_show,
                                        zoom=view_jota['trajectory'][0]['zoom'],
                                        front=view_jota['trajectory'][0]['front'],
                                        lookat=view_jota['trajectory'][0]['lookat'],
                                        up=view_jota['trajectory'][0]['up'],
                                        point_show_normal=False)

    # ------------------------------------------
    # Processing the 3D points to pixels  
    # ------------------------------------------
    
    #print("centros:" + str(centers))
    image = ImageProcessing()
    result = image.loadPointCloud(centers, args.cropped, number)
    
    lista_audio = []
    # ------------------------------------------
    # Better visualization
    # ------------------------------------------
    # Make a more complex open3D window to show object labels on top of 3d
    app = gui.Application.instance
    app.initialize() # create an open3d app

    w = app.create_window("Open3D - 3D Text", 1920, 1080)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  # set black background
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2 * w.scaling
    # Draw entities
    for entity_idx, entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)
    # Draw labels
    for object_idx, object in enumerate(objects):
        label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.15]
        label_text = "Obj: " + object['idx']
        for i, point_cloud in enumerate(list_pcd.values()):
            print("objecto"+str(object["indexed"]))
            print("o i: "+str(i))
            if object['indexed'] == i:
                variable_name = list(list_pcd.keys())[i]
                print("nome da variável:", variable_name)
                label_text += " "+variable_name
                lista_audio.append(variable_name)
        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(255,0,0)
        label.scale = 2
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)
    app.run()
    
   # Inicialize audio processing
    audio = audioprocessing()
    audio_final = audio.loadaudio(lista_audio, number, dimensions)
    
    
if __name__ == "__main__":
    main()