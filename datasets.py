import numpy as np
import torch
from tqdm import trange
from torch.utils.data import Dataset
from obstacles import Obstacle
from utils.config import Config

class BlockageDataset(Dataset):
    def __init__(self, data_num:int, obstacle_ls: list[Obstacle], dtype=torch.float32, cfg: Config = Config.default()):
        super(BlockageDataset, self).__init__()
        self.data_num = data_num

        # Generate ground nodes
        self.gnd_nodes = torch.zeros((data_num, cfg.num_users, 3), dtype=dtype, device=cfg.device)
        for i in trange(data_num):
            gnd_node = []
            while len(gnd_node) < cfg.num_users:
                x = np.random.rand() * cfg.area_size - cfg.area_size // 2
                y = np.random.rand() * cfg.area_size - cfg.area_size // 2
                z = 0
                if (x, y) not in gnd_node:
                    is_inside = any(obstacle.is_inside(x, y, z) for obstacle in obstacle_ls)
                    if not is_inside:
                        gnd_node.append((x, y, z))
            self.gnd_nodes[i] = torch.tensor(np.array(gnd_node), dtype=dtype, device=cfg.device)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.gnd_nodes[idx]

    def to(self, device: torch.device):
        self.gnd_nodes = self.gnd_nodes.to(device)
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