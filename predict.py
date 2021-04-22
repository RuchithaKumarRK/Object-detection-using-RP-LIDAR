"""
    Predict and draw bounding boxes on images using loaded model. 
"""

import torch
import torch.nn as nn
from math import sqrt as sq
from shapely.geometry import box
from shapely.affinity import rotate as rot
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import numpy as np
import cv2
import json
import utils
import imageio
import os
import shutil
import asd
import math
import matplotlib.pyplot as plt

# Load json configs
with open('config.json', 'r') as f:
    config = json.load(f)
boundary = config["boundary"]


def evaluate(model_dir, idx):
    """
        model_dir: path to the file where the model weights are saved (ModelWeights.pth)
        idx: path to the file where the indices are saved (eval.txt), you can use a loop over all indices
    
    """
    root = "D:\LiDAR\Training"
    if os.path.exists("D:\LiDAR\Training\Predict"):
        shutil.rmtree("D:\LiDAR\Training\Predict")
    if ~os.path.exists("D:\LiDAR\Training\Predict"):
        os.mkdir(os.path.join(root, "Predict"))
    APCalc_CPC_Table = np.zeros([5000, 9], dtype = 'object')
    APCalc_IoU_Table = np.zeros([5000, 9], dtype='object')
    Tableupdate = 0
    Tableupdate_IoU = 0
    Number_of_GroundTruthBoxes = 0
    Number_of_PredictedBoxes = 0

    for value in idx:
        Rotated_GroundTruth_box = []
        Rotated_Predicted_box = []
        # Load point cloud data
        if value > 3055:
            break
        print("Working on image " + str(value))
        os.mkdir(os.path.join(root, "Predict", str(value)))
        lidar_file = os.path.join(root, "velodyne", "00" + str(value) + ".bin")
        calibration_file = os.path.join(root, "calib", "00" + str(value) + ".txt")
        label_file = os.path.join(root, "label_2", "00" + str(value) + ".txt")
        calib = utils.calibration(calibration_file)
        target = utils.get_target(label_file, calib)
        model = torch.load(model_dir)  # model that calculates a matrix from input1, so that with get_region_boxes boxes
        # can be calculated
        model.cpu()  # says that the cpu should be used

        a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # raw point cloud
        b = utils.removePoints(a, boundary)  # deletes all points that are not in the boundary
        rgb_map = utils.makeBVFeature(b, boundary, 40 / 512)  # # creates the BEV representation
        # Load trained model and forward, raw input_to_model (512, 1024, 3)
        input_to_model = torch.from_numpy(rgb_map)  # converts the numpy array into a torch tensor
        input_to_model = input_to_model.reshape(1, 3, 512, 1024)  # reshape the tensor, so that he has the right shape

        img = rgb_map.copy()
        img = (img - img.min()) * 255 / (img.max() - img.min())  # normalize the values so that they are between 0 and 255
        img = img.astype(np.uint8)  # change data type
        img2 = img.copy()
        img3 = img.copy()
        img4 = img.copy()
        asd.Tppp(img2)
        Number_of_GroundTruthBoxes += len(target)
        for j in range(len(target)):
            if target[j][1] == 0:
                break
            ground_truth_x = int(target[j][1] * 1024.0)
            ground_truth_y = int(target[j][2] * 512.0)
            ground_truth_width = int(target[j][3] * 1024.0)
            ground_truth_length = int(target[j][4] * 512.0)
            minx = int(ground_truth_x - ground_truth_width / 2)
            miny = int(ground_truth_y - ground_truth_length / 2)
            maxx = int(ground_truth_x + ground_truth_width / 2)
            maxy = int(ground_truth_y + ground_truth_length / 2)
            rotation_angle_GT_arctan2 = (-1) * np.arctan2(target[j][6], target[j][5])
            GT_rect = box(minx, miny, maxx, maxy)
            Rotated_GroundTruth_box.append(Polygon(predbox(rotation_angle_GT_arctan2, GT_rect, "eval_bv_GTBB_rotated.png", img4, root, (102, 0, 0), value)))
        
        # Set model mode to determine whether batch normalization and dropout are engaged
        model.eval()

        output = model(input_to_model.float())

        all_boxes = utils.get_region_boxes(output)  # saves an image of the BEV with the boxes
        Number_of_PredictedBoxes += len(all_boxes)
        print("Number of Ground truth boxes = {GTB}".format(GTB=len(target)))
        print("Number of Predicted boxes = {PB}".format(PB=len(all_boxes)))
        for i in range(len(all_boxes)):
            pred_x = int(all_boxes[i][0] * 1024.0 / 32.0)
            pred_y = int(all_boxes[i][1] * 512.0 / 16.0)
            pred_width = int(all_boxes[i][2] * 1024.0 / 32.0)
            pred_length = int(all_boxes[i][3] * 512.0 / 16.0)
            minx_pred = int(pred_x - pred_width / 2)
            miny_pred = int(pred_y - pred_length / 2)
            maxx_pred = int(pred_x + pred_width / 2)
            maxy_pred = int(pred_y + pred_length / 2)
            cv2.rectangle(img2, (minx_pred, miny_pred), (maxx_pred, maxy_pred), (255, 0, 0), 1)
            imageio.imwrite(os.path.join(root, "Predict", str(value), "eval_bv_pred_norotate.png"), img2)
            Predict_rect = box(minx_pred, miny_pred, maxx_pred, maxy_pred)
            varlist = ["eval_bv_pred_rotrect_sin.png", "eval_bv_pred_rotrect_cos.png", "eval_bv_PredictedBBox_rotated.png", "eval_bv_pred_rotrect_arctan2_funcdef.png", "eval_bv_pred_rotrect_45.png", "eval_bv_pred_rotrect_60.png"]
            RGB = ((0, 0, 0), (255, 153, 51), (0, 0, 0), (0, 102, 51), (255, 255, 0), (102, 0, 0))
            rotation_angle_PredictedBox_arctan = (-1) * np.arctan2(all_boxes[i][5], all_boxes[i][4])
            Rotated_Predicted_box.append(Polygon(predbox(rotation_angle_PredictedBox_arctan, Predict_rect, varlist[2], img4, root, RGB[2], value)))
        if len(all_boxes) != 0:
            if len(target) == 0:
                for tempind_PredBox in range(len(all_boxes)):
                    APTable_Update(APCalc_CPC_Table, Tableupdate, tempind_PredBox, all_boxes, value, 0, 1)
                    Tableupdate += 1
            else:
                Table = np.zeros([len(target), len(all_boxes)])
                for indGT in range(len(target)):
                    for indPB in range(len(all_boxes)):
                        Table[indGT][indPB] = euclidean_dist(int(target[indGT][1] * 1024.0), int(target[indGT][2] * 512.0), int(all_boxes[indPB][0] * 1024.0 / 32.0), int(all_boxes[indPB][1] * 512.0 / 16.0))
                for tempind_PredBox in range(len(all_boxes)):
                        # print (tempind_PredBox)
                        minList = sorted(Table[:, tempind_PredBox])
                        for i in range(0,len(target)):
                            minvalPB = minList[i]
                            rowindex = list(np.where(Table == minvalPB)[0])[0]
                            minvalGT = min(Table[rowindex, :])
                            if minvalPB > 50:
                                APTable_Update(APCalc_CPC_Table, Tableupdate, tempind_PredBox, all_boxes, value, 0, 1)
                                Tableupdate += 1
                                break
                            elif minvalPB == minvalGT:
                                APTable_Update(APCalc_CPC_Table,Tableupdate,tempind_PredBox,all_boxes,value, 1, 0)
                                Tableupdate += 1
                                break
                            else:
                                if len(target) == 1:
                                    if i == 0:
                                        APTable_Update(APCalc_CPC_Table, Tableupdate, tempind_PredBox, all_boxes, value, 0, 1)
                                        Tableupdate += 1
                                        continue
                                elif i == len(target)-1:
                                    APTable_Update(APCalc_CPC_Table, Tableupdate, tempind_PredBox, all_boxes, value, 0, 1)
                                    Tableupdate += 1
                                    continue


        ##########################################


        if len(all_boxes) != 0:
            if len(target) == 0:
                for tempind_PredBox in range(len(all_boxes)):
                    APTable_Update(APCalc_IoU_Table, Tableupdate_IoU, tempind_PredBox, all_boxes, value, 0, 1)
                    Tableupdate_IoU += 1
            else:
                Table_IoU = np.zeros([len(target), len(all_boxes)])
                for indGT in range(len(target)):
                    for indPB in range(len(all_boxes)):
                        Intersected_boxes = Rotated_Predicted_box[indPB].intersection(Rotated_GroundTruth_box[indGT])
                        Intersected_boxes_area = Intersected_boxes.area
                        if(Intersected_boxes_area == 0):
                            Table_IoU[indGT][indPB] = 0
                        else:
                            Union_boxes = Rotated_GroundTruth_box[indGT].union(Rotated_Predicted_box[indPB])
                            Union_boxes_area = Union_boxes.area
                            Table_IoU[indGT][indPB] = Intersected_boxes_area / Union_boxes_area
                for tempind_PredBox in range(len(all_boxes)):
                        maxList = sorted(Table_IoU[:, tempind_PredBox],reverse=True)
                        for i in range(0,len(target)):
                            maxvalPB = maxList[i]
                            rowindex = list(np.where(Table_IoU == maxvalPB)[0])[0]
                            maxvalGT = max(Table_IoU[rowindex, :])
                            if maxvalPB < 0.7:
                                APTable_Update(APCalc_IoU_Table, Tableupdate_IoU, tempind_PredBox, all_boxes, value, 0, 1)
                                Tableupdate_IoU += 1
                                break
                            elif maxvalPB == maxvalGT:
                                APTable_Update(APCalc_IoU_Table,Tableupdate_IoU,tempind_PredBox,all_boxes,value, 1, 0)
                                Tableupdate_IoU += 1
                                break
                            else:
                                if len(target) == 1:
                                    if i == 0:
                                        APTable_Update(APCalc_IoU_Table, Tableupdate_IoU, tempind_PredBox, all_boxes, value, 0, 1)
                                        Tableupdate_IoU += 1
                                        continue
                                elif i == len(target)-1:
                                    APTable_Update(APCalc_IoU_Table, Tableupdate_IoU, tempind_PredBox, all_boxes, value, 0, 1)
                                    Tableupdate_IoU += 1
                                    continue
            if APCalc_CPC_Table[Number_of_PredictedBoxes-1][0] == 0 or APCalc_IoU_Table[Number_of_PredictedBoxes-1][0] == 0:
                print("break")
    APCalc_CPC_Table_sorted = APCalc_CPC_Table[APCalc_CPC_Table[:, 2].argsort()[::-1]]
    APCalc_IoU_Table_sorted = APCalc_IoU_Table[APCalc_IoU_Table[:, 2].argsort()[::-1]]
    for ind in range(len(APCalc_CPC_Table_sorted)):
        if APCalc_CPC_Table_sorted[ind][0] == 0:
            break
        else:
            if ind == 0:
                APCalc_CPC_Table_sorted[ind][5] =  APCalc_CPC_Table_sorted[ind][3]
                APCalc_CPC_Table_sorted[ind][6] = APCalc_CPC_Table_sorted[ind][4]
            else:
                APCalc_CPC_Table_sorted[ind][5] = APCalc_CPC_Table_sorted[ind-1][5] + APCalc_CPC_Table_sorted[ind][3]
                APCalc_CPC_Table_sorted[ind][6] = APCalc_CPC_Table_sorted[ind-1][6] + APCalc_CPC_Table_sorted[ind][4]
            APCalc_CPC_Table_sorted[ind][7] = (APCalc_CPC_Table_sorted[ind][5]) / (APCalc_CPC_Table_sorted[ind][5] + APCalc_CPC_Table_sorted[ind][6])
            APCalc_CPC_Table_sorted[ind][8] = (APCalc_CPC_Table_sorted[ind][5]) / Number_of_GroundTruthBoxes


    APCalc_CPC_Table_sorted = APCalc_CPC_Table_sorted[~np.all(APCalc_CPC_Table_sorted == 0, axis=1)]
    precision_list_CPC = APCalc_CPC_Table_sorted[:, 7]
    recall_list_CPC = APCalc_CPC_Table_sorted[:, 8]
    interpol_points = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precision_interpolated_list_CPC = [precision_interpolated(precision_list_CPC, recall_list_CPC, rec) for rec in interpol_points]
    AP11_CPC = 1 / 11 * sum(precision_interpolated_list_CPC)
    print("Average precision of Center point comparison method using 11 point Interpolation = {ap}".format(ap=AP11_CPC))
    AP_AllPoint_CPC = allpoint_Interpolation(recall_list_CPC, precision_list_CPC)
    print("Average precision of Center point comparison method using All point Interpolation = {ap}".format(ap=AP_AllPoint_CPC))

    for ind2 in range(len(APCalc_IoU_Table_sorted)):
        if APCalc_IoU_Table_sorted[ind2][0] == 0:
            break
        else:
            if ind2 == 0:
                APCalc_IoU_Table_sorted[ind2][5] =  APCalc_IoU_Table_sorted[ind2][3]
                APCalc_IoU_Table_sorted[ind2][6] = APCalc_IoU_Table_sorted[ind2][4]
            else:
                APCalc_IoU_Table_sorted[ind2][5] = APCalc_IoU_Table_sorted[ind2-1][5] + APCalc_IoU_Table_sorted[ind2][3]
                APCalc_IoU_Table_sorted[ind2][6] = APCalc_IoU_Table_sorted[ind2-1][6] + APCalc_IoU_Table_sorted[ind2][4]
            if (APCalc_IoU_Table_sorted[ind2][5]) == 0:
                APCalc_IoU_Table_sorted[ind2][7] = 0
                APCalc_IoU_Table_sorted[ind2][8] = 0
            else:
                APCalc_IoU_Table_sorted[ind2][7] = (APCalc_IoU_Table_sorted[ind2][5]) / (APCalc_IoU_Table_sorted[ind2][5] + APCalc_IoU_Table_sorted[ind2][6])
                APCalc_IoU_Table_sorted[ind2][8] = (APCalc_IoU_Table_sorted[ind2][5]) / Number_of_GroundTruthBoxes
    APCalc_IoU_Table_sorted = APCalc_IoU_Table_sorted[~np.all(APCalc_IoU_Table_sorted == 0, axis=1)]
    precision_list_IoU = APCalc_IoU_Table_sorted[:,7]
    recall_list_IoU = APCalc_IoU_Table_sorted[:,8]
    precision_interpolated_list_IoU = [precision_interpolated(precision_list_IoU, recall_list_IoU, rec) for rec in
                                       interpol_points]
    AP11_IoU = 1 / 11 * sum(precision_interpolated_list_IoU)
    print("Average precision of IoU method using 11 point Interpolation = {ap}".format(ap=AP11_IoU))
    AP_AllPoint_IoU = allpoint_Interpolation(recall_list_IoU, precision_list_IoU)
    print("Average precision of IoU method using All point Interpolation = {ap}".format(
        ap=AP_AllPoint_IoU))



def precision_interpolated(prec_list, rec_list, cur_recall):
    greater_recall = rec_list >= cur_recall
    if greater_recall.sum() == 0:
        return 0
    return max(prec_list[greater_recall])

def allpoint_Interpolation(recall_list, precision_list):
    interpol_points = [0]
    [interpol_points.append(r) for r in recall_list]
    precision_interpolated_list = [0]
    [precision_interpolated_list.append(e) for e in precision_list]
    for i in range(len(precision_interpolated_list) - 1, 0, -1):
        precision_interpolated_list[i - 1] = max(precision_interpolated_list[i - 1], precision_interpolated_list[i])
    ap = 0
    for i in range(1, len(precision_interpolated_list)):
        if interpol_points[i] - interpol_points[i - 1] != 0:
            ap += (interpol_points[i] - interpol_points[i - 1]) * precision_interpolated_list[i]
    return ap
def euclidean_dist(x1, y1, x2, y2):
    dist = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return dist


def APTable_Update(APCalc_Table, Tableupdate, tempind_PredBox , all_boxes, value, TP, FP):
    APCalc_Table[Tableupdate][0] = "Image " + str(value)
    APCalc_Table[Tableupdate][1] = "Predict Box " + str(tempind_PredBox)
    APCalc_Table[Tableupdate][2] = float(all_boxes[tempind_PredBox][6])
    APCalc_Table[Tableupdate][3] = TP
    APCalc_Table[Tableupdate][4] = FP


def predbox(angle, geom, string, img, root, RGB, value):
    pred_rectrotated = rot(geom, angle, use_radians=True)
    #(pred_rectrotated.area)
    pred_rectrotated_coord = list(pred_rectrotated.exterior.coords)
    for jjj in range(len(pred_rectrotated_coord)):
        pred_rectrotated_coord[jjj] = list(pred_rectrotated_coord[jjj])
        for lll in range(len(pred_rectrotated_coord[jjj])):
            pred_rectrotated_coord[jjj][lll] = int(pred_rectrotated_coord[jjj][lll])
    pts = np.array([pred_rectrotated_coord[0], pred_rectrotated_coord[1], pred_rectrotated_coord[2],
                    pred_rectrotated_coord[3]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # img3 = img.copy()
    cv2.polylines(img, [pts], True, RGB)
    imageio.imwrite(os.path.join(root, "Predict", str(value) , string), img)
    return pred_rectrotated_coord


if __name__ == "__main__":
    with open("eval.txt", "r") as f:
        idx = f.readlines()
    idx = [int(line.rstrip("\n")) for line in idx]
    model_dir = 'D:\LiDAR\ModelWeights.pth'

    # Load model, run Predictions and draw boxes
    evaluate(model_dir, idx)
