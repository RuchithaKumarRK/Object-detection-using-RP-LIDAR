# Author - Ruchitha Kumar 
# Lidar Project under guaidance of - Prof. Dr. rer. nat. Stefan Elser

#! /usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import pdb
import math
import sys
import time
import os


def callback(msg):
    tol = 5
    os.system('clear')
    global difference
    global difference1
    global difference2
    # ----------- task_1 : to find min_distance ----------

    no_of_obj = 2
    temp_val = 1
    if temp_val <= no_of_obj:

        dis_of_closest_point = min(msg.ranges)

        # ------------- task_2 : to find angle of closest_point ----------

        index_closest_point = 0
        for index in msg.ranges:
            if index != dis_of_closest_point:
                index_closest_point += 1
            elif index == dis_of_closest_point:
                break

        angle = index_closest_point * msg.angle_increment
        angle_in_degree = np.degrees(angle)  # converting radians to degree

        # --------------------- code to put values in readable file -------------------

        list_of_values = list(msg.ranges)
        filepath = os.path.join('/home/rk/temp', 'range.txt')
        with open(filepath, 'w+') as f:
            for item in list_of_values:
                f.write(" %s\n" % item)
            f.close()

        # --------------------- task_3 : to find length and width ---------------------

        # ---------- rising edge ----------

        index_of_edge_1 = index_closest_point
        index_of_edge_2 = index_of_edge_1
        distance_of_1st_edge = 0
        difference = 0
        tolerance = 0.05
        while difference < tolerance:
            if list_of_values[(index_of_edge_1 + 1) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_1 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_1 + 1) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_1st_edge = index_of_edge_1 % len(msg.ranges)
                    break
                else:
                    index_of_edge_1 += 1
            elif list_of_values[(index_of_edge_1 + 2) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_1 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_1 + 2) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_1st_edge = index_of_edge_1 % len(msg.ranges)
                    break
                else:
                    index_of_edge_1 += 2
            elif list_of_values[(index_of_edge_1 + 3) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_1 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_1 + 3) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_1st_edge = index_of_edge_1 % len(msg.ranges)
                    break
                else:
                    index_of_edge_1 += 3
            elif list_of_values[(index_of_edge_1 + 4) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_1 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_1 + 4) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_1st_edge = index_of_edge_1 % len(msg.ranges)
                    break
                else:
                    index_of_edge_1 += 4
            else:
                distance_of_1st_edge = index_of_edge_1 % len(msg.ranges)
                break

        # ------------ falling edge --------------

        difference = 0
        distance_of_2st_edge = 0
        while difference < tolerance:

            if list_of_values[(index_of_edge_2 - 1) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_2 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_2 - 1) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_2st_edge = index_of_edge_2 % len(msg.ranges)
                    break
                else:
                    index_of_edge_2 -= 1
            elif list_of_values[(index_of_edge_2 - 2) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_2 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_2 - 2) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_2st_edge = index_of_edge_2 % len(msg.ranges)
                    break
                else:
                    index_of_edge_2 -= 2
            elif list_of_values[(index_of_edge_2 - 3) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_2 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_2 - 3) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_2st_edge = index_of_edge_2 % len(msg.ranges)
                    break
                else:
                    index_of_edge_2 -= 3
            elif list_of_values[(index_of_edge_2 - 4) % len(msg.ranges)] != float("inf"):
                difference = abs(list_of_values[(index_of_edge_2 % len(msg.ranges))] - list_of_values[
                    (index_of_edge_2 - 4) % len(msg.ranges)])
                if difference > tolerance:
                    distance_of_2st_edge = index_of_edge_2 % len(msg.ranges)
                    break
                else:
                    index_of_edge_2 -= 4
            else:
                distance_of_2st_edge = index_of_edge_2 % len(msg.ranges)
                break

        # ----------------formulae to find distance---------------------------------------

        distance_of_colsest_point_1 = list_of_values[distance_of_1st_edge % len(msg.ranges)]
        distance_of_colsest_point_2 = list_of_values[distance_of_2st_edge % len(msg.ranges)]

        angle_of_edge_1 = index_of_edge_1 * msg.angle_increment
        angle_of_edge_2 = index_of_edge_2 * msg.angle_increment

        ref_coordinates_of_x = dis_of_closest_point * math.cos(angle)
        ref_coordinates_of_y = dis_of_closest_point * math.sin(angle)

        coordinates_of_edge1_x = distance_of_colsest_point_1 * math.cos(angle_of_edge_1)
        coordinates_of_edge1_y = distance_of_colsest_point_1 * math.sin(angle_of_edge_1)

        coordinates_of_edge2_x = distance_of_colsest_point_2 * math.cos(angle_of_edge_2)
        coordinates_of_edge2_y = distance_of_colsest_point_2 * math.sin(angle_of_edge_2)

        diff_of_refandedge1_x = (ref_coordinates_of_x - coordinates_of_edge1_x) ** 2
        diff_of_refandedge2_x = (ref_coordinates_of_x - coordinates_of_edge2_x) ** 2
        diff_of_refandedge1_y = (ref_coordinates_of_y - coordinates_of_edge1_y) ** 2
        diff_of_refandedge2_y = (ref_coordinates_of_y - coordinates_of_edge2_y) ** 2
        diff_of_edge1andedge2_x = (coordinates_of_edge1_x - coordinates_of_edge2_x) ** 2
        diff_of_edge1andedge2_y = (coordinates_of_edge1_y - coordinates_of_edge2_y) ** 2

        distance_ref_and_edge1 = math.sqrt(diff_of_refandedge1_x + diff_of_refandedge1_y)
        distance_ref_and_edge2 = math.sqrt(diff_of_refandedge2_x + diff_of_refandedge2_y)
        distance_edge1_and_edge2 = math.sqrt(diff_of_edge1andedge2_x + diff_of_edge1andedge2_y)

        # ------------------------------code to detect L or I shape----------------------------
        print (' \n\n')
        print ('------------------------------ 1st object ------------------------------')
        print ('The distance of the closest point = %(dis_of_closest_point * 100)f centimeters' % {
            "dis_of_closest_point "
            "* 100": dis_of_closest_point * 100})
        print ('The angle at which closest point is = %(angle_in_degree)f degrees ' % {"angle_in_degree": angle_in_degree})

        if (distance_of_1st_edge - index_closest_point) % len(msg.ranges) < tol or (index_closest_point -
                                                                                    distance_of_2st_edge) % len(
            msg.ranges) < tol:
            print ('---- :( ---- Not enough points to detect L shape  ---- :( ----')
        elif abs(distance_ref_and_edge1 + distance_ref_and_edge2 - distance_edge1_and_edge2) < 0.01:
            print('---- :o ---- I shape detected ---- :o ----')
            print( 'length or width = %(distance_edge1_and_edge2 * 100)f centimeters' % {
                "distance_edge1_and_edge2 * 100": distance_edge1_and_edge2 * 100})
        else:
            print ('length = %(distance_ref_and_edge1 * 100)f centimeters ' % {
                "distance_ref_and_edge1 * 100": max(distance_ref_and_edge1 * 100, distance_ref_and_edge2 * 100)})
            print( 'width = %(distance_ref_and_edge2 * 100)f centimeters' % {
                "distance_ref_and_edge2 * 100": min(distance_ref_and_edge2 * 100, distance_ref_and_edge2 * 100)})

        # print 'end point 1= {dis_of_closest_point}'.format(dis_of_closest_point=index_of_edge_1 % len(msg.ranges))
        # print 'end point 2= {dis_of_closest_point}'.format(dis_of_closest_point=index_of_edge_2 % len(msg.ranges))
        print ('\n\n')

    temp_val += 1

rospy.init_node('scan_values')
sub = rospy.Subscriber('/scan', LaserScan, callback)
rospy.spin()
