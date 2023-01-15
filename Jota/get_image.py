#!/usr/bin/env python3

import csv
import pickle
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud('/home/jota/Documents/SAVI/Savi_trabalho_2/SaviProject2/Data_scenario/scene.ply')
    print(point_cloud)
    print(np.asarray(point_cloud.points))

    o3d.visualization.draw_geometries([point_cloud],
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    # ------------------------------------------
    # Termination
    # ------------------------------------------

if __name__ == "__main__":
    main()