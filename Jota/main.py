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

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 3.0000000000000004, 3.0000000000000004, 3.83980393409729 ],
			"boundingbox_min" : [ -2.5246021747589111, -1.5300980806350708, -1.4928504228591919 ],
			"field_of_view" : 60.0,
			"front" : [ 0.67118682276199615, -0.70042708582828483, -0.24271412482332699 ],
			"lookat" : [ 0.11570111840900868, -0.033346828483835626, 1.7983664180669476 ],
			"up" : [ -0.73923053281361617, -0.60805019100000812, -0.28950506831651673 ],
			"zoom" : 0.39999999999999969
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def draw_registration_result(source, target, transformation, bbox_to_draw_object, bbox_to_draw_object_target):
    print("correu função")
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

def load_view_point(pcd, filename, id):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.capture_screen_image("image_"+str(id)+".jpg")        
    
    vis.destroy_window()


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

def main():

    # ------------------------------------------
    # Initialization   
    # ------------------------------------------
    p = PointCloudProcessing()
    p.loadPointCloud('/home/jota/Documents/SAVI/savi_22-23/Parte10/data/scene.ply')
    
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud('/home/jota/Documents/SAVI/Savi_trabalho_2/SaviProject2/Data_scenario/scene.ply')

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    # Ex1
    p.preProcess(voxel_size=0.01)

    # Ex2
    p.transform(-108,0,0,0,0,0)
    p.transform(0,0,-37,0,0,0)
    p.transform(0,0,0,-0.85,-1.10,0.35)

    # Create an instance of the class
    pcp = PointCloudProcessing()

    # Set the point cloud as an attribute of the class instance
    pcp.pcd = point_cloud

    # Call the transform function on the class instance, passing in the transformation parameters
    pcp.transform(-108,0,0,0,0,0)
    pcp.transform(0,0,-37,0,0,0)
    pcp.transform(0,0,0,-0.85,-1.10,0.35)
    
    
    
    #Ex3 
    p.crop(-0.9, -0.9, -0.3, 0.9, 0.9, 0.4)

    #Ex4 
    outliers = p.findPlane()
    
    # Ex5 - Clustering
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=60, print_progress=True))
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0,number_of_objects)))

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)
        # Create a dictionary to represent the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color']) # paints the plane in red
        d['center'] = d['points'].get_center()
        objects.append(d) # add the dict of this object to the list

    # Ex6 - ICP
    cereal_box_model = o3d.io.read_point_cloud('../Data_objects/cereal_box_1_1_1.pcd') #0
    cereal_box_model_1 = o3d.io.read_point_cloud('../Data_objects/cereal_box_1_2_1.pcd') #1
    cereal_box_model_2 = o3d.io.read_point_cloud('../Data_objects/cereal_box_5_1_73.pcd') #2
    cereal_box_model_3 = o3d.io.read_point_cloud('../Data_objects/cereal_box_5_2_9.pcd') #3
    bowl_box_model = o3d.io.read_point_cloud('../Data_objects/bowl_2_1_1.pcd') #4
    bowl_box_model_1 = o3d.io.read_point_cloud('../Data_objects/bowl_5_1_66.pcd') #5
    bowl_box_model_2 = o3d.io.read_point_cloud('../Data_objects/bowl_5_1_165.pcd') #6
    soda_can_model = o3d.io.read_point_cloud('../Data_objects/soda_can_2_1_45.pcd') #7
    soda_can_model_1 = o3d.io.read_point_cloud('../Data_objects/soda_can_4_1_94.pcd') #8
    soda_can_model_2 = o3d.io.read_point_cloud('../Data_objects/soda_can_4_1_137.pcd') #9
    coffe_mug_model = o3d.io.read_point_cloud('../Data_objects/coffee_mug_2_1_122.pcd') #10
    coffe_mug_model_1 = o3d.io.read_point_cloud('../Data_objects/coffee_mug_6_1_67.pcd') #11
    

    list_pcd =  [cereal_box_model, cereal_box_model_1,cereal_box_model_2,cereal_box_model_3, bowl_box_model,bowl_box_model_1,bowl_box_model_2, soda_can_model, soda_can_model_1, soda_can_model_2 , coffe_mug_model, coffe_mug_model_1]
    
    for object_idx, object in enumerate(objects):
        object['rmse'] = 10
        object['indexed'] = 100
        min_error = 0.03
        for model_idx, models_object in enumerate(list_pcd): 
            print("Apply point-to-point ICP to object " + str(object['idx']) )

            trans_init = np.asarray([[1, 0, 0, 0],
                                    [0,1,0,0],
                                    [0,0,1,0], 
                                    [0.0, 0.0, 0.0, 1.0]])
            reg_p2p = o3d.pipelines.registration.registration_icp(object['points'], models_object, 1.0, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            #print(str(reg_p2p.inlier_rmse) + "A quando comparado com: " + str(model_idx) )
            #print(reg_p2p)
            
            #######################################################
            ########Comparação####################################
            ######################################################
            bbox_to_draw_object = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
            bbox_to_draw_object.color = (1, 0, 0)
            bbox_to_draw_object_target = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(models_object)
            bbox_to_draw_object_target.color = (0, 1, 0)
            Volume_source = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object)
            Volume_target = o3d.geometry.AxisAlignedBoundingBox.volume(bbox_to_draw_object_target)
            Dimensions_source = o3d.geometry.AxisAlignedBoundingBox.get_extent(bbox_to_draw_object)
            Dimensions_target = o3d.geometry.AxisAlignedBoundingBox.get_extent(bbox_to_draw_object_target)
            #print("Volume da Source: " + str(Volume_source))
            #print("Volume do Target " + str(Volume_target))   
            
            #print("dimensões do Source: " + str(Dimensions_source))
            #print("dimensões do Target: " + str(Dimensions_target))

            volume_compare = abs(Volume_source - Volume_target)  
            print("Volume de comparação " + str(volume_compare))   
            
            x_distance = abs(Dimensions_source[0]- Dimensions_target[0])
            y_distance = abs(Dimensions_source[1]- Dimensions_target[1])
            z_distance = abs(Dimensions_source[2]- Dimensions_target[2])
            
            print("distancia em x: " + str(x_distance))
            print("distancia em y: " + str(y_distance))
            print("distancia em z: " + str(z_distance))
            #if z_distance < 0.01 :  
            
            ######################################################
            ######################################################
            ######################################################
            
            
            #######################################################
            #######################################################
            #######################################################
            
            if  volume_compare < 0.006 :        
                if reg_p2p.inlier_rmse < min_error and reg_p2p.inlier_rmse != 0:
                    if object['rmse'] > reg_p2p.inlier_rmse:
                        object['rmse'] = reg_p2p.inlier_rmse
                        object['indexed'] = model_idx
                        object["fitness"] = reg_p2p.fitness
                        print(object)

                        #draw_registration_result( object['points'],models_object, reg_p2p.transformation, bbox_to_draw_object, bbox_to_draw_object_target)


    # How to classify the object. Use the smallest fitting to decide which object is a "cereal box"
    # minimum_rmse = 10e8 # just a very large number to start
    # cereal_box_object_idx = None
    # bowl_object_idx = None
    # soda_can_idx = None

    # for object_idx, object in enumerate(objects):
    #     # if object['rmse'] < minimum_rmse: # Found a new minimum
    #     #     minimum_rmse = object['rmse']
    #     if object['indexed'] == 0:
    #         #cereal_box_object_idx = object_idx
    #         print("é uma caixa de cereais")
    #     elif object['indexed'] == 1:
    #         #bowl_object_idx = object_idx 
    #         print("é uma taça")
    #     elif object['indexed'] == 2:
    #         #soda_can_idx = object_idx
    #         print("é uma lata de soda") 
            

    #print('The cereal box is object ' + str(cereal_box_object_idx))
    #print('The bowl  is object ' + str(bowl_object_idx))
    
    #print('The soda can is object ' + str(soda_can_idx))
    
    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    # Create a list of entities to draw
    p.inliers.paint_uniform_color([0,1,1]) # paints the plane in red
    entities = []

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox
    
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    
    
    # Draw objects
    for object_idx, object in enumerate(objects):
        # if object_idx == 2: #  show only object idx = 2
        entities.append(object['points'])
        
        
        # Get the oriented bounding box of the point cloud
        bbox_to_draw_object = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(object['points'])
        entities.append(bbox_to_draw_object)
    
        

        
    entities.append(point_cloud)
    
    
    
    for idx, object in enumerate(objects):
        
        save_view_point(object['points'], "viewpoint.json")
        
        load_view_point(point_cloud, "viewpoint.json", idx)     
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(point_cloud)
        # vis.update_geometry(point_cloud)
        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image("image_1.jpg")
        # vis.destroy_window()
    
    
    
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
        #if object_idx == cereal_box_object_idx:
        if object['indexed'] <= 3:
            label_text += ' (Cereal Box)'
        #elif object_idx == bowl_object_idx:
        elif object['indexed'] >= 4 and object['indexed'] <= 6:
            label_text += ' (Bowl) ' 
        #elif object_idx == soda_can_idx:
        elif object['indexed'] >= 7 and object['indexed'] <= 9:
            label_text += " (Soda) "
        elif object['indexed'] >= 10 and object['indexed'] <= 11:
            label_text += " (Mug) "
        else:
            label_text += " (none) "

        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(object['color'][0], object['color'][1],object['color'][2])
        label.scale = 2
        
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()
    # o3d.visualization.draw_geometries(entities,
    #                                 zoom=view['trajectory'][0]['zoom'],
    #                                 front=view['trajectory'][0]['front'],
    #                                 lookat=view['trajectory'][0]['lookat'],
    #                                 up=view['trajectory'][0]['up'],
    #                                 point_show_normal=False)



if __name__ == "__main__":
    main()
