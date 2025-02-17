import numpy as np
import torch
from sympy.physics.units import velocity
from torch import Tensor

from utils.config import Hyperparameters as hp
from datasets import Obstacle

def calc_dist(p1: np.ndarray, p2: np.ndarray, q: np.ndarray):
    v = p2 - p1
    w = q.T - p1
    t = np.clip(np.einsum('xy,y->x', w, v) / np.dot(v, v), 0, 1)
    distances = np.linalg.norm((p1 + t[:, np.newaxis] * v) - q.T, axis=1)
    return distances


def calc_sig_strength(station_pos: np.array, gn_pos: np.ndarray, obst: list[Obstacle]):
    num_gn = gn_pos.shape[0]
    sig = np.zeros(num_gn)

    for i in range(num_gn):
        dist = np.linalg.norm(station_pos - gn_pos[i])

        # Vectorized calculation for minimum distances to obstacles
        min_dist2obst = np.array([np.min(calc_dist(station_pos, gn_pos[i], obst[j].points)) for j in range(len(obst))])

        bk_val = np.tanh(0.2 * np.min(min_dist2obst))
        chan_gain = bk_val * hp.beta_1 / dist + (1 - bk_val) * hp.beta_2 / (dist ** 1.65)
        snr = hp.P_AVG * chan_gain / hp.noise
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

def calc_sig_strength_gpu(station_pos: Tensor, gn_pos: Tensor, obst: Tensor):
    dist = calc_dist_gpu(station_pos, gn_pos, obst)
    bk_val = torch.tanh(torch.min(dist, dim=-1).values*0.2)

    norm = torch.norm(station_pos.unsqueeze(1) - gn_pos.unsqueeze(0), dim=-1)
    chan_gain = bk_val * hp.beta_1 / norm + (1 - bk_val) * hp.beta_2 / (norm ** 1.65)

    snr = hp.P_AVG * chan_gain / hp.noise
    se = torch.log2(1 + snr) # Data rate, Spectral Efficiency
    
    return torch.mean(se, dim=1)

def calc_loss(y_pred: Tensor, x_batch: Tensor, obst_points: Tensor):
    p1, p2, q = y_pred, x_batch, obst_points

    # v와 w의 차원 수정
    v = p2 - p1.unsqueeze(1)  # [batch_size, 4, 3]
    w = q.unsqueeze(0) - p1.unsqueeze(1)  # [batch_size, N_c, 3]

    v_norm_squared = (v ** 2).sum(dim=2, keepdim=True)  # [batch_size, 4, 1]
    dot_product = (v.unsqueeze(2) * w.unsqueeze(1)).sum(dim=3)  # [batch_size, 4, N_c]

    t = torch.clamp(dot_product / v_norm_squared, 0, 1)  # [batch_size, 4, N_c]

    p = p1.unsqueeze(1).unsqueeze(2) + t.unsqueeze(-1) * v.unsqueeze(2)  # [batch_size, 4, N_c, 3]

    dist = torch.norm(p - q.unsqueeze(0).unsqueeze(0), dim=3)  # [batch_size, 4, N_c]

    min_dist2obst = torch.min(dist, dim=2).values  # [batch_size, 4]
    bk_val = torch.tanh(0.2 * min_dist2obst)  # [batch_size, 4]

    norm = torch.norm(v, dim=2)  # [batch_size, 4]
    chan_gain = bk_val * hp.beta_1 / norm + (1 - bk_val) * hp.beta_2 / (norm ** 1.65)  # [batch_size, 4]

    snr = hp.P_AVG * chan_gain / hp.noise  # [batch_size, 4]
    se = torch.log2(1 + snr)  # [batch_size, 4]

    return -torch.mean(se)\

def gn_mobility(init_x, init_y, init_vx, init_vy, num_steps,
                obstacle_ls=None, dt=1.0, alpha=0.9, mu_x=0.0, mu_y=0.0, sigma=1.0, rand_seed=None):
    """
        Gauss-Markov 모빌리티 모델로 2차원 공간의 이동 궤적을 생성.

        매개변수:
        ----------
        init_x, init_y : float
            초기 위치 (x, y)
        init_vx, init_vy : float
            초기 속도 (vx, vy)
        num_steps : int
            시뮬레이션할 총 스텝(시간 단계) 수
        obstacle_ls : list
            장애물 리스트
        dt : float
            한 스텝에서 진행되는 시간 간격 (Δt)
        alpha : float
            모멘텀(기억) 계수 (0 <= alpha <= 1). 클수록 이전 속도의 영향이 큼
        mu_x, mu_y : float
            x,y 축 방향 목표(평균) 속도
        sigma : float
            가우시안 노이즈 표준편차 (속도 변동 폭)
        random_seed : int or None
            난수 시드를 고정할 때 사용. None일 경우 고정 안 함.

        반환값:
        ----------
        positions : np.ndarray
            (num_steps, 2) 형태의 위치(x, y) 배열
        velocities : np.ndarray
            (num_steps, 2) 형태의 속도(vx, vy) 배열
        """
    if obstacle_ls is not None:
        if any(obstacle.is_inside(init_x, init_y, 0) for obstacle in obstacle_ls):
            raise ValueError("Initial position is inside obstacles")

    if rand_seed is not None:
        np.random.seed(rand_seed)

    positions = np.zeros((num_steps,2))
    velocities = np.zeros((num_steps, 2))

    x, y = init_x, init_y
    vx, vy = init_vx, init_vy

    positions[0] = (x, y)
    velocities[0] = (vx, vy)

    for i in range(num_steps-1):
        while True:
            wx = np.random.randn()
            wy = np.random.randn()

            next_vx = alpha * vx + (1 - alpha) * mu_x + np.sqrt(1 - alpha ** 2) * sigma * wx
            next_vy = alpha * vy + (1 - alpha) * mu_y + np.sqrt(1 - alpha ** 2) * sigma * wy

            temp_x = positions[i][0] + next_vx * dt
            temp_y = positions[i][1] + next_vy * dt

            if obstacle_ls is None: break

            if any(obstacle.is_inside(temp_x, temp_y, 0) for obstacle in obstacle_ls):
                continue
            else: break

        positions[i+1] = (temp_x, temp_y)
        velocities[i+1] = (next_vx, next_vy)

    return positions, velocities