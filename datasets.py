from __future__ import annotations

import numpy as np
import torch
from tqdm import trange
from torch.utils.data import Dataset
from obstacles import Obstacle
from utils.config import Config

class BlockageDataset(Dataset):
    """
    A custom dataset for generating blockage data consisting of ground node positions
    that are not located inside any obstacles.

    Attributes:
        data_num (int): Number of ground node configurations (data samples) to generate.
        gnd_nodes (torch.Tensor): A tensor of shape (data_num, num_users, 3) containing
                                  the generated ground node positions.
    """
    def __init__(self, data_num: int, obstacle_ls: list[Obstacle], dtype=torch.float32,
                 cfg: Config = Config.default()):
        """
        Initialize the BlockageDataset.

        Ground nodes are generated randomly within the UAV environment defined by the configuration.
        Each ground node is a 3D coordinate (x, y, z) where z is fixed to 0. The generation ensures that
        nodes are not placed inside any obstacles specified in obstacle_ls.

        :param data_num: Number of data samples to generate.
        :param obstacle_ls: List of Obstacle instances to consider for blockage.
        :param dtype: Data type for the generated ground nodes tensor.
        :param cfg: A Config instance containing settings such as num_users, area_size, and device.
        """
        super(BlockageDataset, self).__init__()
        self.data_num = data_num
        # Initialize the ground nodes tensor with zeros (shape: [data_num, num_users, 3]).
        self.gnd_nodes = torch.zeros((data_num, cfg.num_users, 3), dtype=dtype, device=cfg.device)
        for i in trange(data_num, desc="Generating ground nodes"):
            gnd_node = []
            while len(gnd_node) < cfg.num_users:
                x = np.random.rand() * cfg.area_size - cfg.area_size // 2
                y = np.random.rand() * cfg.area_size - cfg.area_size // 2
                z = 0
                # Only add if the (x, y) is unique and not inside any obstacle.
                if (x, y) not in gnd_node:
                    is_inside = any(obstacle.is_inside(x, y, z) for obstacle in obstacle_ls)
                    if not is_inside:
                        gnd_node.append((x, y, z))
            self.gnd_nodes[i] = torch.tensor(np.array(gnd_node), dtype=dtype, device=cfg.device)

    def __len__(self) -> int:
        """
        :return: The total number of data samples.
        """
        return self.data_num

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve the ground node configuration at the specified index.

        :param idx: Index of the sample to retrieve.
        :return: A tensor of shape (num_users, 3) representing ground node positions.
        """
        return self.gnd_nodes[idx]

    def to(self, device: str) -> 'BlockageDataset':
        """
        Move the ground nodes tensor to the specified device.

        :param device: The target device (e.g., 'cuda' or 'cpu').
        :return: The dataset instance with the tensor moved to the specified device.
        """
        self.gnd_nodes = self.gnd_nodes.to(device)
        return self


class TrainDataset(Dataset):
    """
    A simple dataset for training that wraps an input array into a torch tensor.
    """
    def __init__(self, x, dtype=torch.float32):
        """
        Initialize the TrainDataset.

        :param x: The input data (e.g., a numpy array or list) to be converted into a torch tensor.
        :param dtype: Data type for the tensor.
        """
        self.x = torch.tensor(x, dtype=dtype)

    def __len__(self) -> int:
        """
        :return: The total number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve the data sample at the specified index.

        :param idx: Index of the sample to retrieve.
        :return: A tensor representing the data sample.
        """
        return self.x[idx]

    def to(self, device: str) -> 'TrainDataset':
        """
        Move the data tensor to the specified device.

        :param device: The target device (e.g., 'cuda' or 'cpu').
        :return: The dataset instance with the tensor moved to the specified device.
        """
        self.x = self.x.to(device)
        return self
