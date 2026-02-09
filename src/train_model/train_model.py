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


def _load_shm_copy(shm_name: str, shape: tuple[int, ...]) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        view = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        return np.array(view, copy=True)
    finally:
        shm.close()


def _train_one_experiment(
    cfg: Config,
    x_train: np.ndarray,
    x_val: np.ndarray,
    obst_points: np.ndarray,
    wandb_project: str,
    wandb_name: str,
    desc: str,
    best_model_path: str | None = None,
    final_model_path: str | None = None,
) -> tuple[list[float], list[float]]:
    train_dataset = TrainDataset(x_train, dtype=torch.float32).to(cfg.device)
    val_dataset = TrainDataset(x_val, dtype=torch.float32).to(cfg.device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    obst_tensor = torch.as_tensor(obst_points, dtype=torch.float32, device=cfg.device)

    wandb.init(project=wandb_project, name=wandb_name, config=cfg.to_dict())
    model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_loss = float('inf')

    for epoch in trange(cfg.epochs, desc=desc):
        train_loss = train_pipeline(model, train_dataloader, optimizer, obst_tensor, cfg)
        val_loss = val_pipeline(model, val_dataloader, obst_tensor, cfg)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        if float(val_loss) < best_loss:
            best_loss = float(val_loss)
            if best_model_path:
                createDirectory(os.path.dirname(best_model_path))
                torch.save(model.state_dict(), best_model_path)

        wandb.log({
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch": epoch + 1,
        })

    if final_model_path:
        createDirectory(os.path.dirname(final_model_path))
        torch.save(model.state_dict(), final_model_path)

    wandb.finish()
    return train_losses, val_losses


def _worker_num_gu(
    num_users: int,
    device_id: int,
    base_cfg_dict: dict,
    out_queue,
) -> None:
    cfg_dict = dict(base_cfg_dict)
    cfg_dict['num_users'] = int(num_users)
    cfg = Config(**cfg_dict)
    cfg.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(device_id)

    # Generate data on CPU once per experiment (avoids slow per-sample GPU tensor writes).
    cpu_cfg = cfg.replace(device='cpu')
    set_random_seed(cpu_cfg)
    obstacle_ls, obst_tensor_cpu = create_obstacle_data(cfg=cpu_cfg, return_type='both')

    num_data = int(cpu_cfg.num_samples + cpu_cfg.test_samples)
    x_all = (
        BlockageDataset(num_data, obstacle_ls=obstacle_ls, cfg=cpu_cfg)
        .gnd_nodes[:, :, :2]
        .reshape(-1, cpu_cfg.num_users * 2)
        .cpu()
    )

    x_train_full = x_all[: cpu_cfg.num_samples]
    test_x_xy = x_all[cpu_cfg.num_samples :]

    tmp = test_x_xy.reshape(-1, cpu_cfg.num_users, 2).cpu()
    test_x = (
        torch.cat([tmp, torch.zeros((tmp.shape[0], tmp.shape[1], 1))], dim=2)
        .reshape(-1, cpu_cfg.num_users * 3)
        .cpu()
        .numpy()
    )

    createDirectory(os.path.join(cfg.results_dir, 'data'))
    pd.DataFrame(test_x).to_csv(
        os.path.join(cfg.results_dir, 'data', f'gn_coords_{cpu_cfg.num_users}.csv'),
        index=False,
        header=False,
    )

    x_scaled = cpu_cfg.scaler.transform(x_train_full)
    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=cpu_cfg.random_seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)

    obst_points = obst_tensor_cpu.detach().cpu().numpy().astype(np.float32, copy=False)

    set_random_seed(cfg)
    best_path = os.path.join(cfg.results_dir, 'models', 'num_gu', f'best_num_gu_{cfg.num_users}.pt')
    final_path = os.path.join(cfg.results_dir, 'models', 'num_gu', f'gn_num_{cfg.num_users}_epoch_{cfg.epochs - 1}.pt')
    train_losses, val_losses = _train_one_experiment(
        cfg=cfg,
        x_train=x_train,
        x_val=x_val,
        obst_points=obst_points,
        wandb_project="DL-based UAV Positioning training",
        wandb_name=f"num_gu_training : {cfg.num_users}",
        desc=f"Training with num of gu={cfg.num_users} ({cfg.device})",
        best_model_path=best_path,
        final_model_path=final_path,
    )
    out_queue.put(("num_gu", int(cfg.num_users), train_losses, val_losses))


def _worker_height(
    height: int,
    device_id: int,
    base_cfg_dict: dict,
    x_train_shm: str,
    x_train_shape: tuple[int, ...],
    x_val_shm: str,
    x_val_shape: tuple[int, ...],
    obst_points: np.ndarray,
    out_queue,
) -> None:
    cfg_dict = dict(base_cfg_dict)
    cfg_dict['height'] = int(height)
    cfg = Config(**cfg_dict)
    cfg.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(device_id)

    set_random_seed(cfg)
    x_train = _load_shm_copy(x_train_shm, x_train_shape)
    x_val = _load_shm_copy(x_val_shm, x_val_shape)

    best_path = os.path.join(cfg.results_dir, 'models', 'height', f'best_height_{cfg.height}.pt')
    final_path = os.path.join(cfg.results_dir, 'models', 'height', f'height_{cfg.height}_epoch_{cfg.epochs - 1}.pt')
    train_losses, val_losses = _train_one_experiment(
        cfg=cfg,
        x_train=x_train,
        x_val=x_val,
        obst_points=obst_points,
        wandb_project="DL-based UAV Positioning training",
        wandb_name=f"height_training: {cfg.height}",
        desc=f"Training with height={cfg.height} ({cfg.device})",
        best_model_path=best_path,
        final_model_path=final_path,
    )

    out_queue.put(("height", int(cfg.height), train_losses, val_losses))


def _terminate_all(active: dict[int, mp.Process]) -> None:
    for p in active.values():
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)


def _run_serial(cfg: Config) -> None:
    """Original serial behavior (kept close to prior code)."""
    results = {num_gu: {"train_loss": [], "val_loss": []} for num_gu in cfg.test_list[0]}
    createDirectory(os.path.join(cfg.results_dir, 'data'))
    createDirectory(os.path.join(cfg.results_dir, 'models', 'num_gu'))
    createDirectory(os.path.join(cfg.results_dir, 'models', 'height'))

    for test_cfg in Config.training_gen(mode='num_gu'):
        set_random_seed(test_cfg)
        obstacle_ls, obst_tensor = create_obstacle_data(cfg=test_cfg, return_type='both')
        num_data = test_cfg.num_samples + test_cfg.test_samples
        x = BlockageDataset(num_data,
                            obstacle_ls=obstacle_ls,
                            cfg=test_cfg).gnd_nodes[:, :, :2].reshape(-1, test_cfg.num_users * 2).cpu()
        x, test_x = x[:test_cfg.num_samples], x[test_cfg.num_samples:]

        tmp = test_x.reshape(-1, test_cfg.num_users, 2).cpu()
        test_x = torch.cat([tmp, torch.zeros((tmp.shape[0], tmp.shape[1], 1))], dim=2) \
            .reshape(-1, test_cfg.num_users * 3).cpu().numpy()

        pd.DataFrame(test_x).to_csv(os.path.join(test_cfg.results_dir, 'data', f'gn_coords_{test_cfg.num_users}.csv'),
                                    index=False, header=False)

        x_scaled = test_cfg.scaler.transform(x)

        x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=test_cfg.random_seed)

        train_dataset = TrainDataset(x_train, dtype=torch.float32).to(test_cfg.device)
        val_dataset = TrainDataset(x_val, dtype=torch.float32).to(test_cfg.device)

        train_dataloader = DataLoader(train_dataset, batch_size=test_cfg.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=test_cfg.batch_size, shuffle=False)

        wandb.init(project="DL-based UAV Positioning training", name=f"num_gu_training : {test_cfg.num_users}",
                   config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)

        best_loss = float('inf')

        for epoch in trange(test_cfg.epochs, desc=f"Training with num of gu={test_cfg.num_users}"):
            train_loss = train_pipeline(model, train_dataloader, optimizer, obst_tensor, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.num_users]["train_loss"].append(train_loss)
            results[test_cfg.num_users]["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                            'models', 'num_gu',
                                                            f'best_num_gu_{test_cfg.num_users}.pt'))

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                    'models', 'num_gu',
                                                    f'gn_num_{test_cfg.num_users}_epoch_{test_cfg.epochs - 1}.pt'))
        wandb.finish()

    result_list = []
    for num_gu, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
            result_list.append({"num_gu": num_gu, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    df_results = pd.DataFrame(result_list)
    createDirectory(cfg.results_dir)
    df_results.to_csv(os.path.join(cfg.results_dir, 'num_gu_result.csv'), index=False)
    print("Train complete for Num of GUs.")

    results = {height: {"train_loss": [], "val_loss": []} for height in cfg.test_list[1]}

    set_random_seed(cfg)
    obstacle_ls, obst_tensor = create_obstacle_data(cfg=cfg, return_type='both')
    x = BlockageDataset(cfg.num_samples,
                        obstacle_ls=obstacle_ls,
                        cfg=cfg).gnd_nodes[:, :, :2].reshape(-1, cfg.num_users * 2).cpu()
    x = x[:cfg.num_samples]

    x_scaled = cfg.scaler.transform(x)
    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=cfg.random_seed)

    train_dataset = TrainDataset(x_train, dtype=torch.float32).to(cfg.device)
    val_dataset = TrainDataset(x_val, dtype=torch.float32).to(cfg.device)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    for test_cfg in Config.training_gen(mode='height'):
        wandb.init(project="DL-based UAV Positioning training", name=f"height_training: {test_cfg.height}",
                   config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)

        best_loss = float('inf')

        for epoch in trange(test_cfg.epochs, desc=f"Training with height={test_cfg.height}"):
            train_loss = train_pipeline(model, train_dataloader, optimizer, obst_tensor, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.height]["train_loss"].append(train_loss)
            results[test_cfg.height]["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                            'models', 'height',
                                                            f'best_height_{test_cfg.height}.pt'))

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                    'models', 'height',
                                                    f'height_{test_cfg.height}_epoch_{test_cfg.epochs - 1}.pt'))
        wandb.finish()

    result_list = []
    for height, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
            result_list.append({"height": height, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    df_results = pd.DataFrame(result_list)
    createDirectory(cfg.results_dir)
    df_results.to_csv(os.path.join(cfg.results_dir, 'height_result.csv'), index=False)
    print("Train complete for height.")


def _run_parallel(cfg: Config, devices: list[int]) -> None:
    createDirectory(os.path.join(cfg.results_dir, 'data'))
    createDirectory(os.path.join(cfg.results_dir, 'models', 'num_gu'))
    createDirectory(os.path.join(cfg.results_dir, 'models', 'height'))

    ctx = mp.get_context('spawn')
    base_cfg_dict = cfg.replace(device='cpu').to_dict()

    rr_devices = [int(d) for d in devices]

    # Phase 1: num_gu (dataset generated per experiment)
    out_q = ctx.Queue()
    pending = [int(c.num_users) for c in Config.training_gen(mode='num_gu')]
    active: dict[int, mp.Process] = {}
    received = 0
    total = len(pending)
    num_gu_results = {int(num_gu): {"train_loss": [], "val_loss": []} for num_gu in cfg.test_list[0]}

    def _start_next_num_gu(dev: int) -> None:
        if not pending:
            return
        num_users = pending.pop(0)
        p = ctx.Process(target=_worker_num_gu, args=(int(num_users), int(dev), base_cfg_dict, out_q))
        p.start()
        active[dev] = p
        # Stagger worker creation to avoid simultaneous CUDA context init / memory spikes.
        time.sleep(5.0)

    for dev in rr_devices:
        _start_next_num_gu(dev)

    try:
        while received < total:
            # Drain results first.
            try:
                while True:
                    mode, key, train_losses, val_losses = out_q.get_nowait()
                    if mode == 'num_gu':
                        num_gu_results[int(key)]["train_loss"] = list(train_losses)
                        num_gu_results[int(key)]["val_loss"] = list(val_losses)
                        received += 1
            except queue.Empty:
                pass

            # Reap finished workers and start next on the SAME freed GPU.
            for dev, p in list(active.items()):
                if p.exitcode is None and p.is_alive():
                    continue

                p.join(timeout=1)
                if p.exitcode != 0:
                    raise RuntimeError(f'train_model num_gu worker failed on cuda:{dev}: exitcode={p.exitcode}')

                del active[dev]
                _start_next_num_gu(dev)

            time.sleep(0.2)
    finally:
        _terminate_all(active)

    num_gu_list = []
    for num_gu, res in num_gu_results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
            num_gu_list.append({"num_gu": num_gu, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    pd.DataFrame(num_gu_list).to_csv(os.path.join(cfg.results_dir, 'num_gu_result.csv'), index=False)

    # Phase 2: height (shared dataset)
    cpu_cfg = cfg.replace(device='cpu')
    set_random_seed(cpu_cfg)
    obstacle_ls, obst_tensor_cpu = create_obstacle_data(cfg=cpu_cfg, return_type='both')
    x = (
        BlockageDataset(cpu_cfg.num_samples, obstacle_ls=obstacle_ls, cfg=cpu_cfg)
        .gnd_nodes[:, :, :2]
        .reshape(-1, cpu_cfg.num_users * 2)
        .cpu()
    )
    x_scaled = cpu_cfg.scaler.transform(x)
    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=cpu_cfg.random_seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(x_val, dtype=np.float32)
    obst_points = obst_tensor_cpu.detach().cpu().numpy().astype(np.float32, copy=False)

    shm_train, train_shape = _create_shm_from_ndarray(x_train)
    shm_val, val_shape = _create_shm_from_ndarray(x_val)
    try:
        out_q_h = ctx.Queue()
        pending_h = [int(c.height) for c in Config.training_gen(mode='height')]
        active_h: dict[int, mp.Process] = {}
        received_h = 0
        total_h = len(pending_h)
        height_results = {int(h): {"train_loss": [], "val_loss": []} for h in cfg.test_list[1]}

        def _start_next_height(dev: int) -> None:
            if not pending_h:
                return
            h = pending_h.pop(0)
            p = ctx.Process(
                target=_worker_height,
                args=(
                    int(h),
                    int(dev),
                    base_cfg_dict,
                    shm_train.name,
                    train_shape,
                    shm_val.name,
                    val_shape,
                    obst_points,
                    out_q_h,
                ),
            )
            p.start()
            active_h[dev] = p
            # Stagger worker creation to avoid simultaneous CUDA context init / memory spikes.
            time.sleep(5.0)

        for dev in rr_devices:
            _start_next_height(dev)

        try:
            while received_h < total_h:
                try:
                    while True:
                        mode, key, train_losses, val_losses = out_q_h.get_nowait()
                        if mode == 'height':
                            height_results[int(key)]["train_loss"] = list(train_losses)
                            height_results[int(key)]["val_loss"] = list(val_losses)
                            received_h += 1
                except queue.Empty:
                    pass

                for dev, p in list(active_h.items()):
                    if p.exitcode is None and p.is_alive():
                        continue

                    p.join(timeout=1)
                    if p.exitcode != 0:
                        raise RuntimeError(f'train_model height worker failed on cuda:{dev}: exitcode={p.exitcode}')

                    del active_h[dev]
                    _start_next_height(dev)

                time.sleep(0.2)
        finally:
            _terminate_all(active_h)

        height_list = []
        for height, res in height_results.items():
            for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
                height_list.append({"height": height, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(height_list).to_csv(os.path.join(cfg.results_dir, 'height_result.csv'), index=False)
    finally:
        shm_train.close()
        shm_train.unlink()
        shm_val.close()
        shm_val.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='', help="Comma-separated CUDA device ids, e.g. '0,1,2'. If omitted, run serially.")
    args = parser.parse_args()

    cfg = Config.training()
    devices = _parse_devices(args.devices) if args.devices else []

    if devices:
        _run_parallel(cfg, devices)
    else:
        _run_serial(cfg)
    