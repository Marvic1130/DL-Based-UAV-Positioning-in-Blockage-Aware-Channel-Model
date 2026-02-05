import argparse
import os
import queue
import time
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import trange
import wandb

from datasets import TrainDataset, BlockageDataset
from obstacles import create_obstacle_data
from model import Net
from train import train_pipeline, val_pipeline
from utils.config import Config, set_random_seed
from utils.tools import createDirectory


def _parse_devices(devices: str) -> list[int]:
    return [int(x) for x in devices.split(',') if x.strip()]


def _create_shm_from_ndarray(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, tuple[int, ...]]:
    arr = np.asarray(arr, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shape = tuple(arr.shape)
    np.ndarray(shape, dtype=np.float32, buffer=shm.buf)[:] = arr
    return shm, shape


def _load_shm_slice_copy(shm_name: str, shape: tuple[int, ...]) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        view = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        return np.array(view, copy=True)
    finally:
        shm.close()


def _worker_lr_test(
    lr: float,
    device_id: int,
    cfg_dict: dict,
    x_train_shm: str,
    x_train_shape: tuple[int, ...],
    x_val_shm: str,
    x_val_shape: tuple[int, ...],
    obst_points: np.ndarray,
    out_queue,
) -> None:
    cfg = Config(**cfg_dict)
    cfg.lr = float(lr)
    cfg.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(device_id)

    set_random_seed(cfg)

    x_train = _load_shm_slice_copy(x_train_shm, x_train_shape)
    x_val = _load_shm_slice_copy(x_val_shm, x_val_shape)

    train_dataset = TrainDataset(x_train, dtype=torch.float32).to(cfg.device)
    val_dataset = TrainDataset(x_val, dtype=torch.float32).to(cfg.device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    obst_tensor = torch.as_tensor(obst_points, dtype=torch.float32, device=cfg.device)

    wandb.init(project="DL-based UAV Positioning", name=f"lr_test: {cfg.lr}", config=cfg.to_dict())
    model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in trange(cfg.epochs, desc=f"Training lr={cfg.lr} ({cfg.device})"):
        train_loss = train_pipeline(model, train_dataloader, optimizer, obst_tensor, cfg)
        val_loss = val_pipeline(model, val_dataloader, obst_tensor, cfg)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        wandb.log({
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch": epoch + 1,
        })
    wandb.finish()

    out_queue.put((float(cfg.lr), train_losses, val_losses))


def _run_parallel_lr_test(cfg: Config, devices: list[int]) -> pd.DataFrame:
    # Generate dataset ONCE on CPU to avoid repeating the extremely expensive data generation.
    base_cfg = cfg.replace(device='cpu')
    set_random_seed(base_cfg)
    obstacle_ls, obst_tensor_cpu = create_obstacle_data(cfg=base_cfg, return_type='both')

    x = (
        BlockageDataset(base_cfg.num_samples, obstacle_ls=obstacle_ls, cfg=base_cfg)
        .gnd_nodes[:, :, :2]
        .reshape(-1, base_cfg.num_users * 2)
        .cpu()
    )
    x_scaled = base_cfg.scaler.transform(x)  # NumPy array
    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=base_cfg.random_seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    obst_points = obst_tensor_cpu.detach().cpu().numpy().astype(np.float32, copy=False)

    # Share x_train/x_val via shared memory so each LR worker doesn't copy huge arrays via pickle.
    shm_train, train_shape = _create_shm_from_ndarray(x_train)
    shm_val, val_shape = _create_shm_from_ndarray(x_val)

    ctx = mp.get_context('spawn')
    out_q = ctx.Queue()
    cfg_dict = base_cfg.to_dict()

    # Simple scheduler: at most one active job per GPU.
    pending_lrs = list(cfg.test_list)
    active: dict[int, mp.Process] = {}
    lr_for_dev: dict[int, float] = {}

    def _start_next_on_device(dev: int) -> None:
        if not pending_lrs:
            return
        lr = float(pending_lrs.pop(0))
        p = ctx.Process(
            target=_worker_lr_test,
            args=(
                lr,
                int(dev),
                cfg_dict,
                shm_train.name,
                train_shape,
                shm_val.name,
                val_shape,
                obst_points,
                out_q,
            ),
        )
        p.start()
        active[dev] = p
        lr_for_dev[dev] = lr

    try:
        # Kick off initial wave.
        for dev in devices:
            _start_next_on_device(dev)

        results: dict[float, dict[str, list[float]]] = {float(lr): {"train_loss": [], "val_loss": []} for lr in cfg.test_list}
        finished = 0
        total_jobs = len(cfg.test_list)
        received = 0

        while finished < total_jobs:
            # Harvest finished workers and launch next queued LR on freed GPU.
            for dev, p in list(active.items()):
                if p.exitcode is None and p.is_alive():
                    continue

                p.join(timeout=1)
                if p.exitcode != 0:
                    lr = lr_for_dev.get(dev)
                    raise RuntimeError(f'lr_test worker failed on cuda:{dev} (lr={lr}): exitcode={p.exitcode}')

                del active[dev]
                finished += 1
                _start_next_on_device(dev)

            # Drain result queue.
            drained = False
            try:
                while True:
                    lr, train_losses, val_losses = out_q.get_nowait()
                    results[float(lr)]["train_loss"] = list(train_losses)
                    results[float(lr)]["val_loss"] = list(val_losses)
                    drained = True
                    received += 1
            except queue.Empty:
                if not drained:
                    # Avoid busy-spin.
                    time.sleep(0.2)

        # Ensure we've received all results even if the queue delivery lagged process exit.
        while received < total_jobs:
            lr, train_losses, val_losses = out_q.get(timeout=30.0)
            results[float(lr)]["train_loss"] = list(train_losses)
            results[float(lr)]["val_loss"] = list(val_losses)
            received += 1

        # Build result dataframe.
        result_list = []
        for lr, res in results.items():
            for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
                result_list.append({"lr": lr, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        return pd.DataFrame(result_list)
    finally:
        # Best-effort cleanup.
        for p in active.values():
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)

        shm_train.close()
        shm_train.unlink()
        shm_val.close()
        shm_val.unlink()


def _run_serial_lr_test(cfg: Config) -> pd.DataFrame:
    set_random_seed(cfg)
    obstacle_ls, obst_tensor = create_obstacle_data(cfg=cfg, return_type='both')

    x = (
        BlockageDataset(cfg.num_samples, obstacle_ls=obstacle_ls, cfg=cfg)
        .gnd_nodes[:, :, :2]
        .reshape(-1, cfg.num_users * 2)
        .cpu()
    )
    x_scaled = cfg.scaler.transform(x)

    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=cfg.random_seed)

    train_dataset = TrainDataset(x_train, dtype=torch.float32).to(cfg.device)
    val_dataset = TrainDataset(x_val, dtype=torch.float32).to(cfg.device)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    results = {lr: {"train_loss": [], "val_loss": []} for lr in cfg.test_list}
    for test_cfg in Config.lr_test_gen():
        wandb.init(project="DL-based UAV Positioning", name=f"lr_test: {test_cfg.lr}", config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)

        for epoch in trange(test_cfg.epochs, desc=f"Training with lr={test_cfg.lr}"):
            train_loss = train_pipeline(model, train_dataloader, optimizer, obst_tensor, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.lr]["train_loss"].append(train_loss)
            results[test_cfg.lr]["val_loss"].append(val_loss)

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        wandb.finish()

    result_list = []
    for lr, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
            result_list.append({"lr": lr, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    return pd.DataFrame(result_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='', help="Comma-separated CUDA device ids, e.g. '0,1,2'. If omitted, run serially.")
    args = parser.parse_args()

    cfg = Config.lr_test()

    devices = _parse_devices(args.devices) if args.devices else []
    if devices:
        df_results = _run_parallel_lr_test(cfg, devices)
    else:
        df_results = _run_serial_lr_test(cfg)

    createDirectory(cfg.results_dir)
    df_results.to_csv(os.path.join(cfg.results_dir, 'result.csv'), index=False)
    print("Training complete.")
    print(df_results)