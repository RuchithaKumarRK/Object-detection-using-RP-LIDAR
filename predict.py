"""
    Predict and draw bounding boxes on images using loaded model. 
"""
import numpy
import shapely.geometry
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import utils
import imageio
from math import sqrt
import os
from PIL import Image

# Load json configs
with open('config.json', 'r') as f:
    config = json.load(f)
boundary = config["boundary"]


def edist(x1, x2, y1, y2):
    distance = numpy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def eval(model_dir, idx):
    """
        model_dir: path to the file where the model weights are saved (ModelWeights.pth)
        idx: path to the file where the indices are saved (eval.txt), you can use a loop over all indices
    """

    for idx in idx:
        if idx > 3001:
            break
        root = "F:\Training"
        # Load point cloud data
        # lidar_file = os.path.join(root, "Velodyne", "00" + str(idx) + ".bin")
        lidar_file = os.path.join(root, "Velodyne", "003001.bin")
        # path to one lidar file with an indice from idx (e.g. Path/to/file/velodyne/003001.bin)
        # calibration_file = os.path.join(root, "calib", "00" + str(idx) + ".txt")
        calibration_file = os.path.join(root, "calib", "003001.txt")
        # path to one calibration file with an indice from idx (e.g. Path/to/file/calib/003001.txt)
        # label_file = os.path.join(root, "label_2", "00" + str(idx) + ".txt")
        label_file = os.path.join(root, "label_2", "003001.txt")
        # path to the ground truth label file with an indice from idx (e.g. Path/to/file/label_2/003001.txt)

        calib = utils.calibration(calibration_file)
        target = utils.get_target(label_file, calib)

        model = torch.load(
            model_dir)  # model that calculates a matrix from input, so that with get_region_boxes boxes can be calculated
        model.cpu()  # says that the cpu should be used
        a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # raw point cloud
        b = utils.removePoints(a, boundary)  # deletes all points that are not in the boundary
        rgb_map = utils.makeBVFeature(b, boundary, 40 / 512)  # # creates the BEV represantation

        # Load trained model and forward, raw input (512, 1024, 3)

        input = torch.from_numpy(rgb_map)  # convertes the numpy array into a torch tensor
        input = input.reshape(1, 3, 512, 1024)  # reshape the tensor, so that he has the right shape
        img = rgb_map.copy()
        img = (img - img.min()) * 255 / (
                img.max() - img.min())  # normalize the values so that they are between 0 and 255
        img = img.astype(np.uint8)  # change data type

        for j in range(len(target)):
            if target[j][1] == 0:
                break

            ground_truth_x = int(target[j][1] * 1024.0)
            ground_truth_y = int(target[j][2] * 512.0)
            ground_truth_width = int(target[j][3] * 1024.0)
            ground_truth_length = int(target[j][4] * 512.0)
            rect_top1 = int(ground_truth_x - ground_truth_width / 2)
            rect_top2 = int(ground_truth_y - ground_truth_length / 2)
            rect_bottom1 = int(ground_truth_x + ground_truth_width / 2)
            rect_bottom2 = int(ground_truth_y + ground_truth_length / 2)
            # cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255),1)

            # Here the boxes are drawn without the rotation
            # For your evaluation with IoU you could write
            # a class for rotated rectangles.
            # For this the library shapely and especially
            # shapely.geometry.box (creating a box)
            # shapely.affinity.rotate and shapely.affinity.translate
            # (rotating and translating) the box
            # Also the intersection of two such rectangles can be calculated
            # pretty easy.
            # For the comparison of the box center this is actual not
            # needed, but can also be used

            # Example:

            angle = -1 * np.arctan2(target[j][6], target[j][5])
            gt_bbox = shapely.geometry.box(rect_top1, rect_top2, rect_bottom1, rect_bottom2)
            gt_bbox = shapely.affinity.rotate(gt_bbox, angle, use_radians=True)
            corners = gt_bbox.exterior.coords[:]

            for j in range(len(corners)):
                cv2.line(img, (int(corners[j][0]), int(corners[j][1])),
                         (int(corners[(j + 1) % len(corners)][0]), int(corners[(j + 1) % len(corners)][1])), (0, 0, 0),
                         1)

        # Set model mode to determine whether batch normalization and dropout are engaged

        model.eval()
        output = model(input.float())
        all_boxes = utils.get_region_boxes(output)  # saves an image of the BEV with the boxes
        imageio.imwrite('F:\Training\Predict\eval_bv.png', img)

        for i in range(len(all_boxes)):
            print("Box predicted!")
            pred_x = int(all_boxes[i][0] * 1024.0 / 32.0)
            pred_y = int(all_boxes[i][1] * 512.0 / 16.0)
            pred_width = int(all_boxes[i][2] * 1024.0 / 32.0)
            pred_length = int(all_boxes[i][3] * 512.0 / 16.0)
            rect_top3 = int(pred_x - pred_width / 2)
            rect_top4 = int(pred_y - pred_length / 2)
            rect_bottom3 = int(pred_x + pred_width / 2)
            rect_bottom4 = int(pred_y + pred_length / 2)
            # cv2.rectangle(img, (rect_top3, rect_top4), (rect_bottom3, rect_bottom4), (0, 255, 0), 1)

            angle = -1 * np.arctan2(all_boxes[i][5], all_boxes[i][4])
            gt_bbox1 = shapely.geometry.box(rect_top3, rect_top4, rect_bottom3, rect_bottom4)
            gt_bbox1 = shapely.affinity.rotate(gt_bbox1, angle, use_radians=True)
            cornerss = gt_bbox1.exterior.coords[:]

            for i in range(len(cornerss)):
                cv2.line(img, (int(cornerss[i][0]), int(cornerss[i][1])),
                         (int(cornerss[(i + 1) % len(cornerss)][0]), int(cornerss[(i + 1) % len(cornerss)][1])),
                         (0, 255, 0), 1)

        imageio.imwrite('F:\Training\Predict\eval_bvp.png', img)
        # img = Image.open('F:\Training\Predict\eval_bv.png')
        img = Image.open('F:\Training\Predict\eval_bvp.png')
        img.show()

        # mid_point1 = (rect_bottom1, rect_bottom2)
        # mid_point2 = ((pred_x + pred_width/2), (pred_y + pred_length/2))

        # -------------cpcomp-------------

        array = np.zeros((10, 10))
        for j in range(len(target)):
            for i in range(len(all_boxes)):
                array[j][i] = edist(int(target[j][1] * 1024.0), int(all_boxes[i][0] * 1024.0 / 32.0),
                                    int(target[j][2] * 512.0), int(all_boxes[i][1] * 512.0 / 16.0))

        for j in range(len(target)):
            for i in range(len(all_boxes)):
                if array.all() < 50:
                    array[j][i] = min(array[j][i])
                    for i in range(len(all_boxes)):
                        for j in range(len(target)):
                            array[i][j] = min(array[i][j])
        if array[j][i] == array[i][j]:
            print(array)


if __name__ == "__main__":
    with open("eval.txt", "r") as f:
        idx = f.readlines()
    idx = [int(line.rstrip("\n")) for line in idx]
    model_dir = 'F:/ModelWeights.pth'

    # Load model, run predictions and draw boxes

    eval(model_dir, idx)
