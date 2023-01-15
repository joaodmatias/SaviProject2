#!/usr/bin/env python3


from copy import deepcopy
import math
import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

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


    # ------------------------------------------
    # Execution
    # ------------------------------------------
    
    pc = PointCloudProcessing()
    
    print("Load a ply point cloud, print it, and render it")
    pc.loadPointCloud('Scenes/pc/01.ply')
    point_cloud = o3d.io.read_point_cloud('/home/jota/Documents/SAVI/Savi_trabalho_2/SaviProject2/Data_scenario/scene.ply')

    pc.preProcess(0.02)

    # Set coordinate frame
    pc.transform(250,0,0,0,0,0)
    pc.transform(0,0,45,0,0,0)
    pc.transform(0,0,0,1.5,-0.45,0.25)
    
    # Create an instance of the class
    pcp = PointCloudProcessing()

    # Set the point cloud as an attribute of the class instance
    pcp.pcd = point_cloud

    # Call the transform function on the class instance, passing in the transformation parameters
    pcp.transform(250,0,0,0,0,0)
    pcp.transform(0,0,45,0,0,0)
    pcp.transform(0,0,0,1.5,-0.45,0.25)
    
    
    pc.crop(0, 0, -0.1, 1, 1, 0.25)

    outliers = pc.findPlane()
    
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.03, min_points=10, print_progress=True))
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
        d['idx'] = str(object_idx + 1)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color']) 
        d['center'] = d['points'].get_center()
        objects.append(d) # add the dict of this object to the list

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    # Create a list of entities to draw
    pc.inliers.paint_uniform_color([0,1,1]) # paints the plane in red
    entities = []

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(pc.bbox)
    entities.append(bbox_to_draw)
    
    

    # Draw objects
    for object_idx, object in enumerate(objects):
        entities.append(object['points'])
        color = object['color'] * 255
        print('Object ' + object['idx'] + ' has ' + str(color) + ' color')

    entities.append(point_cloud)

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