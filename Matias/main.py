#!/usr/bin/env python3

from copy import deepcopy
import open3d as o3d
import numpy as np
from more_itertools import locate
import webcolors
import matplotlib.pyplot as plt
from matplotlib import cm

from Classes import *

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.0, 1.0, 1.0 ],
			"boundingbox_min" : [ -0.059999999999999998, -0.059999999999999998, -0.19050790332514628 ],
			"field_of_view" : 60.0,
			"front" : [ 0.87337729543889542, 0.37839459939229664, 0.30664250677716548 ],
			"lookat" : [ -1.7729291118608874, -0.23816766922699825, -0.66915049497169721 ],
			"up" : [ -0.22849970590836063, -0.23766517292413558, 0.94408852867659254 ],
			"zoom" : 2.0
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    pc = PointCloudProcessing()
    
    print("Load a ply point cloud, print it, and render it")
    scenario_name = '10'
    point_cloud = pc.loadPointCloud('Scenes/pc/' + scenario_name + '.ply')
  

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
    entities = [table]

    entities.append(frame)

    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pc.bbox)
    entities.append(bbox_to_draw)

    # Draw objects
    for object_idx, object in enumerate(objects):
        entities.append(object['points'])

        properties = ObjectProperties(object)
        size = properties.getSize()
        print("This object's SIZE is " + str(size))

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

    entities.append(pc.pcd)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])


    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()