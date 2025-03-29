import numpy as np
import torch
import plotly.graph_objects as go
from abc import abstractmethod
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from utils.config import Config


class Obstacle:
    """
    Abstract base class representing a generic obstacle.

    :param x: The x-coordinate of the obstacle.
    :param y: The y-coordinate of the obstacle.
    :param height: The height of the obstacle.
    """
    def __init__(self, x: int, y: int, height: int):
        self.shape = {"x": x, "y": y, "height": height}
        self.points = [0, 0, 0]
        self.mesh = None

    @property
    def x(self) -> int:
        """
        :return: The x-coordinate of the obstacle.
        """
        return self.shape["x"]

    @property
    def y(self) -> int:
        """
        :return: The y-coordinate of the obstacle.
        """
        return self.shape["y"]

    @property
    def height(self) -> int:
        """
        :return: The height of the obstacle.
        """
        return self.shape["height"]

    def __str__(self) -> str:
        """
        :return: A string describing the obstacle's shape.
        """
        return f"DotCloud: {self.shape}"

    def plot(self, ax: plt.Axes) -> PathCollection:
        """
        Plot the obstacle points using matplotlib.

        :param ax: A matplotlib Axes object on which to plot the points.
        :return: The scatter plot object.
        """
        return ax.scatter(self.points[0], self.points[1], self.points[2])

    @abstractmethod
    def plotly_obj(self) -> go.Mesh3d:
        """
        Create and return a Plotly 3D mesh object representing the obstacle.

        :return: A Plotly Mesh3d object.
        """
        pass

    @abstractmethod
    def is_inside(self, x: float, y: float, z: float) -> bool:
        """
        Determine if a given point is inside the obstacle.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param z: z-coordinate of the point.
        :return: True if the point is inside the obstacle, otherwise False.
        """
        pass

    def to_torch(self, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Convert the obstacle's points to a PyTorch tensor.

        :param device: The torch.device on which to allocate the tensor.
        :param dtype: Data type for the tensor (default: torch.float32).
        """
        self.points = torch.tensor(self.points, dtype=dtype).to(device)


class CubeObstacle(Obstacle):
    """
    Represents a cube-shaped obstacle.

    :param x: x-coordinate of the cube's starting point.
    :param y: y-coordinate of the cube's starting point.
    :param height: Height of the cube.
    :param width: Width of the cube.
    :param depth: Depth of the cube.
    :param dot_num: Density factor for generating points on faces (default: 0.05).
    """
    def __init__(self, x: int, y: int, height: int, width: int, depth: int, dot_num: float = 0.05):
        super().__init__(x, y, height)
        self.shape["width"] = width
        self.shape["depth"] = depth

        top = int(width * depth * dot_num)
        fb = int(width * height * dot_num)
        lr = int(depth * height * dot_num)

        __points = [
            # Front face points.
            np.array([
                x + width * np.random.rand(fb),
                np.full(fb, y),
                height * np.random.rand(fb)
            ]),
            # Back face points.
            np.array([
                x + width * np.random.rand(fb),
                np.full(fb, y + depth),
                height * np.random.rand(fb)
            ]),
            # Left face points.
            np.array([
                np.full(lr, x),
                y + depth * np.random.rand(lr),
                height * np.random.rand(lr)
            ]),
            # Right face points.
            np.array([
                np.full(lr, x + width),
                y + depth * np.random.rand(lr),
                height * np.random.rand(lr)
            ]),
            # Top face points.
            np.array([
                x + width * np.random.rand(top),
                y + depth * np.random.rand(top),
                np.full(top, height)
            ])
        ]
        self.points = np.concatenate(__points, axis=1)
        self.plotly_obj()

    @property
    def width(self) -> int:
        """
        :return: The width of the cube.
        """
        return self.shape["width"]

    @property
    def depth(self) -> int:
        """
        :return: The depth of the cube.
        """
        return self.shape["depth"]

    def __str__(self) -> str:
        """
        :return: A string describing the cube's shape parameters.
        """
        return f"CubeCloud: {self.shape}"

    def plotly_obj(self, opacity=1, color=None) -> go.Mesh3d:
        """
        Create and return a Plotly Mesh3d object representing the cube.

        :param opacity: Opacity of the mesh (default: 1).
        :param color: Color of the mesh (default: None).
        :return: A Plotly Mesh3d object.
        """
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
            (0, 1, 2), (0, 2, 3),  # Bottom face.
            (4, 5, 6), (4, 6, 7),  # Top face.
            (0, 1, 5), (0, 5, 4),  # Front face.
            (1, 2, 6), (1, 6, 5),  # Right face.
            (2, 3, 7), (2, 7, 6),  # Back face.
            (3, 0, 4), (3, 4, 7)   # Left face.
        ]
        i = [t[0] for t in triangles]
        j = [t[1] for t in triangles]
        k = [t[2] for t in triangles]

        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        z_coords = [v[2] for v in vertices]

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

    def is_inside(self, x: float, y: float, z: float) -> bool:
        """
        Check if the given point is inside the cube.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param z: z-coordinate of the point.
        :return: True if the point is inside the cube, otherwise False.
        """
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.depth and
                0 <= z <= self.height)


class CylinderObstacle(Obstacle):
    """
    Represents a cylindrical obstacle.

    :param x: x-coordinate of the cylinder's base.
    :param y: y-coordinate of the cylinder's base.
    :param height: Height of the cylinder.
    :param radius: Radius of the cylinder.
    :param dot_num: Density factor for generating points (default: 0.05).
    """
    def __init__(self, x: int, y: int, height: int, radius: int, dot_num: float = 0.05):
        super().__init__(x, y, height)
        self.shape["radius"] = radius

        t_num = int(radius**2 * np.pi * dot_num)
        s_num = int(2 * radius * np.pi * height * dot_num)

        r_top = radius * np.sqrt(np.random.rand(t_num))
        theta_top = np.random.rand(t_num) * 2 * np.pi
        angles_side = np.linspace(0, 2 * np.pi, s_num, endpoint=False)
        __points = [
            # Top face points.
            np.array([
                x + r_top * np.cos(theta_top),
                y + r_top * np.sin(theta_top),
                np.full(t_num, height)
            ]),
            # Side face points.
            np.array([
                x + radius * np.cos(angles_side),
                y + radius * np.sin(angles_side),
                height * np.random.rand(s_num)
            ])
        ]
        self.points = np.concatenate(__points, axis=1)
        self.plotly_obj()

    @property
    def radius(self) -> int:
        """
        :return: The radius of the cylinder.
        """
        return self.shape["radius"]

    def __str__(self) -> str:
        """
        :return: A string describing the cylinder's parameters.
        """
        return f"CylinderCloud: {self.shape}"

    def plotly_obj(self, opacity=1, color=None, n=100) -> go.Mesh3d:
        """
        Create and return a Plotly Mesh3d object representing the cylinder.

        :param opacity: Opacity of the mesh (default: 1).
        :param color: Color of the mesh (default: None).
        :param n: Number of segments to approximate the circle (default: 100).
        :return: A Plotly Mesh3d object.
        """
        if (self.mesh is not None and
                self.mesh.get('theta_num') == n and
                (color is None or color == self.mesh['obj'].color) and
                opacity == self.mesh['obj'].opacity):
            return self.mesh['obj']

        vertices = []
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Generate vertices for the bottom circle.
        for t in theta:
            vertices.append([self.x + self.radius * np.cos(t),
                             self.y + self.radius * np.sin(t),
                             0])
        # Generate vertices for the top circle.
        for t in theta:
            vertices.append([self.x + self.radius * np.cos(t),
                             self.y + self.radius * np.sin(t),
                             self.height])

        bottom_idx = len(vertices)
        vertices.append([self.x, self.y, 0])
        top_idx = len(vertices)
        vertices.append([self.x, self.y, self.height])

        i_side, j_side, k_side = [], [], []
        i_bottom, j_bottom, k_bottom = [], [], []
        i_top, j_top, k_top = [], [], []

        for i in range(n):
            next_i = (i + 1) % n

            # Side faces (two triangles per segment).
            i_side.append(i)
            j_side.append(next_i)
            k_side.append(n + i)

            i_side.append(next_i)
            j_side.append(n + next_i)
            k_side.append(n + i)

            # Bottom face.
            i_bottom.append(bottom_idx)
            j_bottom.append(i)
            k_bottom.append(next_i)

            # Top face.
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

    def is_inside(self, x: float, y: float, z: float) -> bool:
        """
        Check if the given point is inside the cylinder.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param z: z-coordinate of the point.
        :return: True if the point is within the circular base and between 0 and the cylinder's height, otherwise False.
        """
        return ((self.x - x)**2 + (self.y - y)**2 <= self.radius**2 and
                0 <= z <= self.height)


def create_obstacle_data(dot_num: float = 0.05, device: torch.device = Config.default().device,
                         return_type: str = "both"):
    """
    Create obstacle data.

    :param dot_num: Density factor for generating points on faces (default: 0.05).
    :param device: The torch.device on which to allocate the obstacle points tensor (default: CPU).
    :param return_type: Specifies what to return:
                        "list" returns only the list of obstacle instances,
                        "tensor" returns only the obstacle points tensor,
                        "both" returns a tuple of (obstacle list, obstacle tensor).
                        (default: "both")
    :return: Depending on return_type, returns either the obstacle list, the tensor, or both.
    """
    obstacle_ls = [
        CubeObstacle(-30, 35, 40, 80, 40, dot_num=dot_num),
        CubeObstacle(70, -75, 45, 20, 75, dot_num=dot_num),
        CubeObstacle(-30, -75, 35, 80, 20, dot_num=dot_num),
        CubeObstacle(-75, -75, 35, 20, 150, dot_num=dot_num),
        CubeObstacle(70, 35, 35, 20, 40, dot_num=dot_num),
        CubeObstacle(30, -40, 55, 20, 60, dot_num=dot_num),
        CylinderObstacle(-5, -10, 70, 20, dot_num=dot_num)
    ]

    obst_points = torch.cat([torch.tensor(obstacle.points, dtype=torch.float32)
                               for obstacle in obstacle_ls], dim=1).mT.to(device)

    return {
        "list": obstacle_ls,
        "tensor": obst_points,
        "both": (obstacle_ls, obst_points)
    }[return_type]