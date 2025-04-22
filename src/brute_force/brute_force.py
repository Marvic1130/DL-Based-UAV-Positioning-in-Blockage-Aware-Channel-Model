# brute_force.py
import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config, set_random_seed
from datasets import BlockageDataset
from obstacles import create_obstacle_data
from utils.tools import calc_sig_strength_gpu, createDirectory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------- #
# 공통 유틸
# --------------------------------------------------------------- #
def save_df(path: str, filename: str, data: list, num_users: int) -> None:
    """결과 CSV를 (존재 시 append) 저장한다."""
    # 동적 컬럼 이름
    cols = []
    for i in range(num_users):
        cols += [f"gnd{i+1}_x", f"gnd{i+1}_y", f"gnd{i+1}_z"]
    cols += ["result_x", "result_y", "result_z"]

    createDirectory(path)
    full = os.path.join(path, filename)

    new_df = pd.DataFrame(data, columns=cols)
    if os.path.exists(full):
        base = pd.read_csv(full)
        new_df = pd.concat([base, new_df], ignore_index=True)

    new_df.to_csv(full, index=False)
    logging.info(" ✔ saved → %s", full)


# --------------------------------------------------------------- #
# 브루트포스 실행
# --------------------------------------------------------------- #
def run_brute_force(test_cfg: Config) -> None:
    """주어진 설정(test_cfg)으로 브루트포스 1회 수행"""
    set_random_seed(test_cfg)

    logging.info("◆ Brute‑Force | #users=%d  height=%.1f  grid_step=%.2f",
                 test_cfg.num_users, test_cfg.height, test_cfg.grid_step)

    # 장애물 / 데이터셋 준비
    obst_list = create_obstacle_data(dot_num=0.1,
                                     cfg=test_cfg,
                                     return_type="list")

    dataset = BlockageDataset(                 # ❶ 후보 UAV 위치 그리드 포함
        data_num       = 10_000,
        obstacle_ls    = obst_list,
        dtype          = torch.float32,
        cfg            = test_cfg
    )

    # ── build candidate UAV‑position grid ──────────────────────────
    side_pts   = int(round(test_cfg.area_size / test_cfg.grid_step)) + 1
    xs         = torch.linspace(-test_cfg.area_size / 2,
                                test_cfg.area_size / 2,
                                side_pts,
                                device=test_cfg.device)
    ys         = xs.clone()
    X, Y       = torch.meshgrid(xs, ys, indexing="ij")
    station_pos = torch.stack(
        [X.reshape(-1),
         Y.reshape(-1),
         torch.full_like(X.reshape(-1), test_cfg.height)],
        dim=1)                                    # [N,3]

    grid_side  = side_pts

    # ── obstacle point cloud (K,3) tensor ─────────────────────────
    obst_pts = torch.cat(
        [torch.tensor(o.points,
                      dtype=torch.float32,
                      device=test_cfg.device)
         for o in obst_list],
        dim=1
    ).T                                          # [K,3]

    # results accumulator
    results: list[list[float]] = []

    # ── ground‑node 좌표 CSV를 로드해 dataset.gnd_nodes 교체 ────────────
    csv_name   = f"gn_coords_{test_cfg.num_users}.csv"
    csv_path   = os.path.join(Config.training().results_dir, "data", csv_name)
    gnd_array  = pd.read_csv(csv_path, header=None).values
    dataset.gnd_nodes = torch.tensor(
        gnd_array, dtype=torch.float32, device=test_cfg.device
    ).reshape(-1, test_cfg.num_users, 3)

    # -- DataLoader : larger batch & pinned memory -------------------------
    loader = DataLoader(dataset,
                        batch_size=1,            # <= can tune to VRAM
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0)

    # ---------------------------------------------------------------------
    with torch.no_grad():                         # inference only
        for gnd_nodes in tqdm(loader, desc="Searching", miniters=20):
            gnd_nodes = gnd_nodes.squeeze(0)                  # [U,3]

            # ---- 신호 세기 계산 (chunk 처리) ----------------------------
            sig_chunks = []
            for s in range(0, station_pos.size(0), test_cfg.chunk):
                sig_chunks.append(
                    calc_sig_strength_gpu(station_pos[s:s+test_cfg.chunk],
                                          gnd_nodes,
                                          obst_pts,
                                          cfg=test_cfg)
                )
            sig_all = torch.cat(sig_chunks, dim=0).reshape(grid_side, grid_side)

            # ---- 최적 좌표 계산 ----------------------------------------
            max_idx  = torch.unravel_index(torch.argmax(sig_all), sig_all.shape)
            x_max    = max_idx[0].item()*test_cfg.grid_step - (test_cfg.area_size//2)
            y_max    = max_idx[1].item()*test_cfg.grid_step - (test_cfg.area_size//2)
            z_max    = test_cfg.height

            results.append(list(gnd_nodes.cpu().numpy().ravel()) + [x_max, y_max, z_max])
            if test_cfg.device == "cuda":
                torch.cuda.synchronize()

    # 결과 저장
    save_df(
        path      = test_cfg.results_dir,
        filename  = f"mode_{test_cfg.mode}_U{test_cfg.num_users}_H{test_cfg.height}.csv",
        data      = results,
        num_users = test_cfg.num_users
    )


# --------------------------------------------------------------- #
# 메인 루틴
# --------------------------------------------------------------- #
if __name__ == "__main__":
    # ① 사용자 수별 브루트포스 --------------------------------------------------
    for cfg_num in Config.brute_force_gen(mode="num_gu"):
        run_brute_force(cfg_num)

    # ② 고도별 브루트포스 -------------------------------------------------------
    for cfg_h in Config.brute_force_gen(mode="height"):
        run_brute_force(cfg_h)

    logging.info("All brute‑force sweeps finished.")