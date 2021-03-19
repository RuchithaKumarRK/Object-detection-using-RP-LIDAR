"""
    Util scripts for building features, fetching ground truths, etc.
    All of this functions are only one way to do the task, you can use your own
    functions as well
"""

import torch
from torch.autograd import Variable
import numpy as np
import json
import math

# Load configs from json
with open('config.json', 'r') as f:
    config = json.load(f)

# Classes of objects and anchors
class_list = config ["class_list"]
#anchors = config ["anchors"]

def removePoints(PointCloud, BoundaryCond):
    """
        Here you will remove those points that are not of interest.
        Same function as in converter.py
    """
    # Boundary condition
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ'] ; maxZ = BoundaryCond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = np.where(
            (PointCloud[:, 0] >= minX) & 
            (PointCloud[:, 0] <= maxX) & 
            (PointCloud[:, 1] >= minY) & 
            (PointCloud[:, 1] <= maxY) & 
            (PointCloud[:, 2] >= minZ) & 
            (PointCloud[:, 2] <= maxZ)
            )
    PointCloud = PointCloud[mask]
    return PointCloud
    
def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    """
        Here you will converte the KITTI Point Cloud into a data format used by the algorithms. 
        The output should be a 512 x 1024 x 3 numpy array,
        in the 3 channels should be the density, the heightand the intensity stored.
        Same function as in converter.py
    """    
    # 1024 x 1024 x 3
    Height = 1024 + 1
    Width = 1024 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width / 2)
    
    # sort-3times
    indices = np.lexsort((-PointCloud[:,2], PointCloud[:,1], PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis = 0, return_index = True)
    PointCloud_frac = PointCloud[indices]
    
    # Some important problem is image coordinate is (y,x), not (x,y)
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis = 0, return_index = True, return_counts = True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts
    RGB_Map = np.zeros((Height,Width, 3))

    # RGB channels respectively
    RGB_Map[:,:,0] = densityMap
    RGB_Map[:,:,1] = heightMap
    RGB_Map[:,:,2] = intensityMap
    
    save = np.zeros((512, 1024, 3))
    save = RGB_Map[0:512, 0:1024, :]
    return save

def get_target(label_file, calib):
    """
        Here you will read the KITTI labels and converte them, such that they fit the input of the model.
        Make sure that you only have objects with class "Car"
        label_file: path to the label file
        calib: calibration class
        Output: Array with n rows and 7 collums
                n = number of objects in that label file (only those objects that class is in class list 
                (see config.json)
                0. collum: Class (here 0 = Car, because this is our only class of interest)
                1, 2. collum: location (first x axis, second y axis)
                3. collum: object width 
                4. collum: object length
                5, 6. collum: Real and imaginary part of the rotation. To get the angle you can use np.arctan2
    """    
    target = np.zeros([50, 7], dtype = np.float32)
    with open(label_file, 'r') as f:
        lines = f.readlines() 
    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class in class_list:

            # Get target 3D object location x, y
            t_lidar = calib.box3d_cam_to_velo(obj[8:])
            location_x = t_lidar[0][0]          
            location_y = t_lidar[0][1]            
            
            if (location_x > 0) & (location_x < 40) & (location_y > -40) & (location_y < 40):
                
                # Make sure target inside the covering area (0,1)
                target[index][2] = t_lidar[0][0] / 40
                
                # Should put this in [0,1] ,so divide max_size 80 m
                target[index][1] = (t_lidar[0][1]+40) / 80
                obj_width = obj[9].strip()
                obj_length = obj[10].strip()
                target[index][3] = float(obj_width) / 80
                target[index][4] = float(obj_length) / 40

                # Get target Observation angle of object, ranging [-pi .. pi]
                obj_alpha = obj[14].strip()

                # Im axis
                target[index][5] = math.sin(float(obj_alpha))
                # Re axis
                target[index][6] = math.cos(float(obj_alpha))
                for i in range(len(class_list)):
                    if obj_class == class_list[i]:
                        target[index][0] = i
                index = index + 1
    target = target[0:index]
    return target

class calibration():
    """
        Calibration class
    """
    def __init__(self, calib_file):
        with open(calib_file) as fi:
            lines = fi.readlines()
            assert (len(lines) == 8)
        obj = lines[2].strip().split(' ')[1:]
        self.P2 = np.array(obj, dtype=np.float32)
        self.P2 = self.P2.reshape(3,4)
        obj = lines[4].strip().split(' ')[1:]
        self.R0 = np.array(obj, dtype=np.float32)
        self.R0 = self.R0.reshape(3,3)
        obj = lines[5].strip().split(' ')[1:]
        self.Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        self.Tr_velo_to_cam = self.Tr_velo_to_cam.reshape(3,4)
        
    
    def box3d_cam_to_velo(self, box3d):
        """
            Function that transforms a point from the camera coordinatesystem into
            the lidar coordinatesystem
        """
        def project_cam2velo(cam):
            T = np.zeros([4, 4], dtype = np.float32)
            T[:3, :] = self.Tr_velo_to_cam
            T[3, 3] = 1
            T_inv = np.linalg.inv(T)
            lidar_loc_ = np.dot(T_inv, cam)
            lidar_loc = lidar_loc_[:3]
            return lidar_loc.reshape(1, 3)

        _, _, _, tx, ty, tz, ry = [float(i) for i in box3d]
        cam = np.ones([4, 1])
        cam[0] = tx
        cam[1] = ty
        cam[2] = tz
        t_lidar = project_cam2velo(cam)
        return t_lidar
    
def get_region_boxes(output_model):
    """
        This function calculates from the model output the predicted boxes.
        Input: output_model Matrix of dimension: 1 x 75 x 16 x 32. Is calculated from the model
        Output: List of length = number of predicted boxes. Every list item is a list of length 7
        0. item: x value - between 0 and 32 so we have to divide it by 32 and multiply it with 1024,
                           so that it fits our image
        1. item: y value - between 0 and 16 so we have to divide it by 16 and multiply it with 512,
                           so that it fits our image
        2. item: object width -  between 0 and 32 so we have to divide it by 32 and multiply it with 1024,
                           so that it fits our image
        3. item: object length - between 0 and 16 so we have to divide it by 16 and multiply it with 512,
                           so that it fits our image
        4., 5. item: imaginary and real part of the rotation, the angle of the rotation can be calculated with np.arctan2
        6. Confidence of the prediction - this is the score which will be needed for the AP
    """
    conf_thresh = 0.5 # threshold for the predicted confidence
    
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] # anchor for the dimension 
    num_anchors = 5                                                           # of the object ( 5 anchors )
    anchor_step = int(len(anchors) / num_anchors)
    
    if output_model.dim() == 3:
        output_model = output_model.unsqueeze(0)
    batch = output_model.size(0)
    assert(output_model.size(1) == 15 * num_anchors)
    
    # h: 16, w: 32
    h = output_model.size(2)
    w = output_model.size(3)
    nB = output_model.data.size(0)

    nA = num_anchors
    nH = output_model.data.size(2)
    nW = output_model.data.size(3)
    anchor_step = int(len(anchors) / num_anchors)
    output_model = output_model.view(nB, nA, 15, nH, nW)
    
    x = torch.sigmoid(output_model.index_select(2, Variable(torch.LongTensor([0]))).view(nB, nA, nH, nW))
    y = torch.sigmoid(output_model.index_select(2, Variable(torch.LongTensor([1]))).view(nB, nA, nH, nW))
    w = output_model.index_select(2, Variable(torch.LongTensor([2]))).view(nB, nA, nH, nW)
    l = output_model.index_select(2, Variable(torch.LongTensor([3]))).view(nB, nA, nH, nW)
    im = output_model.index_select(2, Variable(torch.LongTensor([4]))).view(nB, nA, nH, nW)
    re = output_model.index_select(2, Variable(torch.LongTensor([5]))).view(nB, nA, nH, nW)
    conf = torch.sigmoid(output_model.index_select(2, Variable(torch.LongTensor([6]))).view(nB, nA, nH, nW))
    
    
    pred_boxes = torch.FloatTensor(7, nB * nA * nH * nW)
    grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)
    grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)
    anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_l = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1]))
    
    
    anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
    anchor_l = anchor_l.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)

    pred_boxes[0] = x.data.view(nB * nA * nH * nW) + grid_x
    pred_boxes[1] = y.data.view(nB * nA * nH * nW) + grid_y
    pred_boxes[2] = torch.exp(w.data).view(nB * nA * nH * nW) * anchor_w
    pred_boxes[3] = torch.exp(l.data).view(nB * nA * nH * nW) * anchor_l
    pred_boxes[4] = im.data.view(nB * nA * nH * nW)
    pred_boxes[5] = re.data.view(nB * nA * nH * nW)
    
    pred_boxes[6] = conf.data.view(nB * nA * nH * nW)
    pred_boxes = convert(pred_boxes.transpose(0, 1).contiguous().view(-1, 7))   #torch.Size([2560, 15])
    
    all_boxes =[]
    for i in range(2560): # 2560 = 5 * 16 * 32 = num_anchors * h * w
        if pred_boxes[i][6] > conf_thresh:
            all_boxes.append(pred_boxes[i])
    return all_boxes
   
def convert(matrix):
    return torch.FloatTensor(matrix.size()).copy_(matrix)
    
