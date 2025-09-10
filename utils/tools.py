import os
from typing import List
import numpy as np
import torch
from torch import Tensor
from utils.config import Config
from obstacles import Obstacle, CubeObstacle, CylinderObstacle

def calc_dist(p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
    v = p2 - p1
    w = q.T - p1
    t = np.clip(np.einsum('xy,y->x', w, v) / np.dot(v, v), 0, 1)
    distances = np.linalg.norm((p1 + t[:, np.newaxis] * v) - q.T, axis=1)
    return distances


def calc_sig_strength(station_pos: np.array, gn_pos: np.ndarray, obst: List[Obstacle], cfg: Config = Config.default()):
    num_gn = gn_pos.shape[0]
    sig = np.zeros(num_gn)

    for i in range(num_gn):
        dist = np.linalg.norm(station_pos - gn_pos[i])

        # Vectorized calculation for minimum distances to obstacles
        min_dist2obst = np.array([np.min(calc_dist(station_pos, gn_pos[i], obst[j].points)) for j in range(len(obst))])

        bk_val = np.tanh(0.2 * np.min(min_dist2obst))
        chan_gain = bk_val * cfg.beta_1 / dist + (1 - bk_val) * cfg.beta_2 / (dist ** 1.65)
        snr = cfg.power * chan_gain / cfg.noise
        se = np.log2(1 + snr)
        sig[i] = se

    return np.mean(sig)

def calc_dist_gpu(p1: Tensor, p2: Tensor, q: Tensor):
    v = p2[None, :, :] - p1[:, None, :]
    w = q[None, :, :] - p1[:, None, :]
    v_norm_squared = (v ** 2).sum(dim=2, keepdim=True)
    dot_product = (v[:, :, None, :] * w[:, None, :, :]).sum(dim=3)
    t = torch.clamp(dot_product / v_norm_squared, 0, 1)
    p = p1[:, None, None, :] + t[..., None] * v[:, :, None, :]
    dist = torch.norm(p - q[None, None, :, :], dim=3)
    return dist

def calc_sig_strength_gpu(station_pos: Tensor, gn_pos: Tensor, obst: Tensor, cfg: Config=Config.default()):
    dist = calc_dist_gpu(station_pos, gn_pos, obst)
    bk_val = torch.tanh(torch.min(dist, dim=-1).values*0.2)

    norm = torch.norm(station_pos.unsqueeze(1) - gn_pos.unsqueeze(0), dim=-1)
    chan_gain = bk_val * cfg.beta_1 / norm + (1 - bk_val) * cfg.beta_2 / (norm ** 1.65)

    snr = cfg.power * chan_gain / cfg.noise
    se = torch.log2(1 + snr) # Data rate, Spectral Efficiency
    
    return torch.mean(se, dim=1)

def calc_loss(y_pred: Tensor, x_batch: Tensor, obst_points: Tensor, cfg: Config=Config.default()):
    p1, p2, q = y_pred, x_batch, obst_points

    # v와 w의 차원 수정
    v = p2 - p1.unsqueeze(1)  # [batch_size, gn_num, 3]
    w = q.unsqueeze(0) - p1.unsqueeze(1)  # [batch_size, N_c, 3]

    v_norm_squared = (v ** 2).sum(dim=2, keepdim=True)  # [batch_size, gn_num, 1]
    dot_product = (v.unsqueeze(2) * w.unsqueeze(1)).sum(dim=3)  # [batch_size, gn_num, N_c]

    t = torch.clamp(dot_product / v_norm_squared, 0, 1)  # [batch_size, gn_num, N_c]

    p = p1.unsqueeze(1).unsqueeze(2) + t.unsqueeze(-1) * v.unsqueeze(2)  # [batch_size, gn_num, N_c, 3]

    dist = torch.norm(p - q.unsqueeze(0).unsqueeze(0), dim=3)  # [batch_size, gn_num, N_c]

    min_dist2obst = torch.min(dist, dim=2).values  # [batch_size, gn_num]
    bk_val = torch.tanh(0.2 * min_dist2obst)  # [batch_size, gn_num]

    norm = torch.norm(v, dim=2)  # [batch_size, 4]
    chan_gain = bk_val * cfg.beta_1 / norm + (1 - bk_val) * cfg.beta_2 / (norm ** 1.65)  # [batch_size, gn_num]

    snr = cfg.power * chan_gain / cfg.noise  # [batch_size, gn_num]
    se = torch.log2(1 + snr)  # [batch_size, gn_num]

    return -torch.mean(se)

def create_mask(obstacle_ls: list, grid: torch.Tensor) -> torch.Tensor:

    X = grid[..., 0]
    Y = grid[..., 1]
    Z = grid[..., 2]

    mask_grid = torch.ones_like(X, dtype=torch.int8, device=X.device)

    for obstacle in obstacle_ls:
        mask = None
        if isinstance(obstacle, CubeObstacle):
            mask_x = (obstacle.x <= X) & (X <= obstacle.x + obstacle.width)
            mask_y = (obstacle.y <= Y) & (Y <= obstacle.y + obstacle.depth)
            mask_z = (0 <= Z) & (Z <= obstacle.height)
            mask = mask_x & mask_y & mask_z

        elif isinstance(obstacle, CylinderObstacle):
            dist_sq = (obstacle.x - X)**2 + (obstacle.y - Y)**2
            mask_base = dist_sq <= obstacle.radius**2
            mask_z = (0 <= Z) & (Z <= obstacle.height)
            mask = mask_base & mask_z

        if mask is not None:
            mask_grid[mask] = 0

    return mask_grid

def probabilistic_channel_model(gn_tensor: Tensor, obst_ls: List[Obstacle], a_1: float = 11.95, a_2: float = 0.14,
                                cfg: Config=Config.default()):
    """
    :param gn_tensor: gn_tensor size: (gn_num, 3)
    :param obst_ls: list of Obstacle
    :param a_1: a_1 parameter
    :param a_2: a_2 parameter
    :param chunk_size: chunk size
    :param cfg: configuration

    :return: se: spectral efficiency (n, m) shape matrix
    """
    x = torch.arange(-100, 100.01, cfg.grid_step, device=cfg.device)
    y = torch.arange(-100, 100.01, cfg.grid_step, device=cfg.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.full_like(X, cfg.height, device=cfg.device)
    grid = torch.stack([X, Y, Z], dim=-1)
    n, m, _ = grid.shape

    results = []

    for i in range(0, n, cfg.chunk):
        grid_chunk = grid[i: i + cfg.chunk]
        grid_chunk = grid_chunk.unsqueeze(2)  # 이제 shape: (chunk_size, m, 1, 3)
        gn_expanded = gn_tensor.unsqueeze(0).unsqueeze(0)

        diff = gn_expanded - grid_chunk
        dist = torch.norm(diff, dim=-1)  # shape: (chunk_size, m, 4)
        horizontal_dist = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # shape: (chunk_size, m, 4)
        tanval = (180 / torch.pi) * torch.atan(diff[..., 2] / horizontal_dist)  # shape: (chunk_size, m, 4)

        P_LOS = 1 / (1 + a_1 * torch.exp(-a_2 * (tanval - a_1)))  # shape: (chunk_size, m, 4)
        chan_gain = P_LOS * cfg.beta_1 / dist + (1 - P_LOS) * cfg.beta_2 / (dist ** 1.65)

        snr = cfg.power * chan_gain / cfg.noise
        se = torch.log2(1 + snr)  # shape: (chunk_size, m, 4)

        se_mean = se.mean(dim=-1)
        results.append(se_mean.cpu())

    result_se = torch.cat(results, dim=0)

    mask = create_mask(obst_ls, grid)
    result_se = result_se * mask

    return result_se

def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)