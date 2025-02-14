import logging
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import torch
from numpy import dtype
from tqdm import trange
from torch.utils.data import Dataset

from utils.config import Hyperparameters as hp

class Obstacle:
    def __init__(self, x: int, y: int, height: int):
        self.shape = {"x": x,
                      "y": y,
                      "height": height}
        self.points = [0,0,0]

    @property
    def x(self):
        return self.shape["x"]

    @property
    def y(self):
        return self.shape["y"]

    @property
    def height(self):
        return self.shape["height"]

    def __str__(self):
        return f"DotCloud: {self.shape}"

    def plot(self, ax: plt.Axes):
        return ax.scatter(self.points[0], self.points[1], self.points[2])

    @abstractmethod
    def is_inside(self, x: float, y: float, z: float):
        pass

    def to_torch(self, device: torch.device, dtype: torch.dtype = torch.float32):
        self.points =  torch.tensor(self.points, dtype=dtype).to(device)


class CubeObstacle(Obstacle):
    def __init__(self, x: int, y: int, height: int, width: int, depth: int, dot_num: float = 0.05):
        super().__init__(x, y, height)
        self.shape["width"] = width
        self.shape["depth"] = depth

        top = int(width*depth*dot_num)
        fb = int(width*height*dot_num)
        lr = int(depth*height*dot_num)

        __points = [
                    # front face
                    np.array([x + width * np.random.rand(fb),
                              [y] * np.ones(fb, ),
                              height * np.random.rand(fb)]),
                    # back face
                    np.array([x + width * np.random.rand(fb),
                              [(y + depth)] * np.ones(fb, ),
                              height * np.random.rand(fb)]),
                    # left face
                    np.array([[x] * np.ones(lr, ),
                              y + depth * np.random.rand(lr),
                              height * np.random.rand(lr)]),
                    # right face
                    np.array([[x + width] * np.ones(lr, ),
                              y + depth * np.random.rand(lr),
                              height * np.random.rand(lr)]),
                    # top face
                    np.array([x + width * np.random.rand(top),
                              y + depth * np.random.rand(top),
                              [height] * np.ones(top, )])
                    ]
        # concatenate all
        self.points = np.concatenate(__points, axis=1)

    @property
    def width(self):
        return self.shape["width"]

    @property
    def depth(self):
        return self.shape["depth"]

    def __str__(self):
        return f"CubeCloud: {self.shape}"

    def is_inside(self, x: float, y: float, z: float):
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.depth and
                self.height <= z <= self.height)



class CylinderObstacle(Obstacle):
    def __init__(self, x: int, y: int, height: int, radius: int, dot_num: float = 0.05):
        super().__init__(x, y, height)
        self.shape["radius"] = radius

        t_num = int(radius**2*np.pi*dot_num)
        s_num = int(2*radius*np.pi*height*dot_num)

        r_top = radius * np.sqrt(np.random.rand(t_num))
        theta_top = np.random.rand(t_num) * 2 * np.pi
        angles_side = np.linspace(0, 2 * np.pi, s_num, endpoint=False)
        __points = [
            # top face
            np.array([x + r_top * np.cos(theta_top),
                      y + r_top * np.sin(theta_top),
                      [height] * t_num]),
            # side faces
            np.array([x + radius * np.cos(angles_side),
                      y + radius * np.sin(angles_side),
                      height * np.random.rand(s_num)])
        ]
        self.points = np.concatenate(__points, axis=1)

    @property
    def radius(self):
        return self.shape["radius"]

    def __str__(self):
        return f"CylinderCloud: {self.shape}"

    def is_inside(self, x: float, y: float, z: float):
        return ((self.x - x)**2 + (self.y - y)**2 <= self.radius**2 and
                self.height <= z <= self.height)


class BlockageDataset(Dataset):
    def __init__(self, data_num:int, obstacle_ls: list[Obstacle], gnd_num: int = 4, dtype=torch.float32):
        super(BlockageDataset, self).__init__()
        self.data_num = data_num

        # Generate station positions
        X, Y = np.meshgrid(
            np.arange(-hp.area_size // 2, hp.area_size // 2),
            np.arange(-hp.area_size // 2, hp.area_size // 2),
            indexing='xy'
        )
        Z = np.full_like(X, 70)
        self.station_pos = torch.tensor(np.stack((X, Y, Z), axis=-1).reshape(-1, 3), dtype=dtype)

        # Generate ground nodes
        self.gnd_nodes = torch.zeros((data_num, gnd_num, 3), dtype=dtype)
        for i in trange(data_num):
            gnd_node = []
            while len(gnd_node) < gnd_num:
                x = np.random.rand() * hp.area_size - hp.area_size // 2
                y = np.random.rand() * hp.area_size - hp.area_size // 2
                z = 0
                if (x, y) not in gnd_node:
                    is_inside = any(obstacle.is_inside(x, y, z) for obstacle in obstacle_ls)
                    if not is_inside:
                        gnd_node.append((x, y, z))
            self.gnd_nodes[i] = torch.tensor(np.array(gnd_node), dtype=dtype)

        # obstacle points
        obst_points = []
        for obstacle in obstacle_ls:
            obst_points.append(torch.tensor(obstacle.points, dtype=dtype))
        self.obst_points = torch.cat([op for op in obst_points], dim=1).mT

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.station_pos, self.gnd_nodes[idx], self.obst_points

    def to(self, device: torch.device):
        self.station_pos = self.station_pos.to(device)
        self.gnd_nodes = self.gnd_nodes.to(device)
        self.obst_points = self.obst_points.to(device)
        return self
    
    
class TrainDataset(Dataset):
    def __init__(self, x, dtype=torch.float32):
        self.x = torch.tensor(x, dtype=dtype)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        return self

if __name__ == "__main__":
    cube = CubeObstacle(0, 0, 0, 10, 10)
    print(cube.shape)
    print(cube.points)
