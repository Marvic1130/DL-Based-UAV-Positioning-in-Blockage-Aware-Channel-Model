import logging
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
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
        self.mesh = None

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
    def plotly_obj(self):
        pass

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
        self.plotly_obj()

    @property
    def width(self):
        return self.shape["width"]

    @property
    def depth(self):
        return self.shape["depth"]

    def __str__(self):
        return f"CubeCloud: {self.shape}"

    def plotly_obj(self, opacity=1, color=None):
        if self.mesh is not None and (color is None or color == self.mesh.color) and opacity == self.mesh.opacity:
            return self.mesh

        vertices = [
            [self.x, self.y, 0],
            [self.x + self.width, self.y, 0],
            [self.x + self.width, self.y + self.depth, 0],
            [self.x, self.y + self.depth, 0],
            [self.x, self.y, self.height],
            [self.x + self.width, self.y, self.height],
            [self.x + self.width, self.y + self.depth, self.height],
            [self.x, self.y + self.depth, self.height]
        ]

        triangles = [
            # 바닥면
            (0, 1, 2), (0, 2, 3),
            # 천정면
            (4, 5, 6), (4, 6, 7),
            # 앞면
            (0, 1, 5), (0, 5, 4),
            # 오른쪽 면
            (1, 2, 6), (1, 6, 5),
            # 뒷면
            (2, 3, 7), (2, 7, 6),
            # 왼쪽 면
            (3, 0, 4), (3, 4, 7)
        ]
        i = [t[0] for t in triangles]
        j = [t[1] for t in triangles]
        k = [t[2] for t in triangles]

        # 각 꼭짓점 좌표 분리
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]

        # Mesh3d 트레이스 생성 (Plotly는 기본적으로 원근법 적용)
        self.mesh = go.Mesh3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            i=i,
            j=j,
            k=k,
            opacity=opacity,
            color=color,
            flatshading=True
        )
        return self.mesh

    def is_inside(self, x: float, y: float, z: float):
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.depth and
                0 <= z <= self.height)



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
        self.plotly_obj()

    @property
    def radius(self):
        return self.shape["radius"]

    def __str__(self):
        return f"CylinderCloud: {self.shape}"

    def plotly_obj(self, opacity=1, color=None, n=100):
        if (self.mesh is not None and
                self.mesh['theta_num'] == n and
                (color is None or color == self.mesh['obj'].color) and
                opacity == self.mesh['obj'].opacity):
            return self.mesh['obj']

        vertices = []
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)

        for t in theta:
            vertices.append([self.x + self.radius * np.cos(t), self.y + self.radius * np.sin(t), 0])
        for t in theta:
            vertices.append([self.x + self.radius * np.cos(t), self.y + self.radius * np.sin(t), self.height])

        bottom_idx = len(vertices)
        vertices.append([self.x, self.y, 0])
        top_idx = len(vertices)
        vertices.append([self.x, self.y, self.height])

        i_side, j_side, k_side = [], [], []
        i_bottom, j_bottom, k_bottom = [], [], []
        i_top, j_top, k_top = [], [], []

        for i in range(n):
            next_i = (i + 1) % n

            # 측면: 두 개의 삼각형으로 구성
            i_side.append(i)
            j_side.append(next_i)
            k_side.append(n + i)

            i_side.append(next_i)
            j_side.append(n + next_i)
            k_side.append(n + i)

            # 바닥면: 중심 점과 인접 두 점으로 삼각형 생성
            i_bottom.append(bottom_idx)
            j_bottom.append(i)
            k_bottom.append(next_i)

            # 천정면: 중심 점과 인접 두 점으로 삼각형 생성 (정확한 법선 방향을 위해 순서 반대로)
            i_top.append(top_idx)
            j_top.append(n + next_i)
            k_top.append(n + i)

        i_total = i_side + i_bottom + i_top
        j_total = j_side + j_bottom + j_top
        k_total = k_side + k_bottom + k_top

        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]

        self.mesh = {"obj": go.Mesh3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            i=i_total,
            j=j_total,
            k=k_total,
            opacity=opacity,
            color=color,
            flatshading=True,
            name='Cylinder'
        ), 'theta_num': n}
        return self.mesh['obj']

    def is_inside(self, x: float, y: float, z: float):
        return ((self.x - x)**2 + (self.y - y)**2 <= self.radius**2 and
                0 <= z <= self.height)


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
