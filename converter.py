import numpy as np
import imageio
import json

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    
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
    
def removePoints(PointCloud, BoundaryCond):
    
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
    
if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    boundary = config["boundary"]
    Discretization = 40 / 512
    PointCloud = np. fromfile("/home/berens/remote/PointRCNN/data/KITTI/object/training/velodyne/000010.bin", dtype = np.float32).reshape(-1,4)
    PointCloud = removePoints(PointCloud, boundary)
    rgb_map = makeBVFeature(PointCloud, boundary, 40 / 512)
    imageio.imwrite('input_visualization.png', rgb_map)
