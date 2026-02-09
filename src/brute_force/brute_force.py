import os
import argparse
import glob
import logging
import math
import sys
import queue
import time
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from obstacles import create_obstacle_data
from utils.config import Config, set_random_seed
from utils.tools import calc_sig_strength_gpu, createDirectory


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _result_columns(num_users: int) -> list[str]:
    cols = []
    for i in range(num_users):
        cols += [f'gnd{i + 1}_x', f'gnd{i + 1}_y', f'gnd{i + 1}_z']
    cols += ['result_x', 'result_y', 'result_z']
    return cols


def _build_station_grid(cfg: Config) -> tuple[torch.Tensor, int]:
    side_pts = int(round(cfg.area_size / cfg.grid_step)) + 1
    xs = torch.linspace(-cfg.area_size / 2, cfg.area_size / 2, side_pts, device=cfg.device)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    station_pos = torch.stack(
        [X.reshape(-1), Y.reshape(-1), torch.full_like(X.reshape(-1), cfg.height)],
        dim=1,
    )
    return station_pos, side_pts


def _cfg_spawn_payload(cfg: Config) -> tuple[dict, dict]:
    """Return (dataclass_fields_dict, extras_dict) safe for mp spawn.

    Config.brute_force_gen() injects non-dataclass attributes (e.g. grid_step/chunk/mode)
    via Config.replace(). Those are not included in Config.to_dict()/asdict(), so we pass
    them separately for worker reconstruction.
    """
    base = cfg.to_dict()
    extras = {}
    for key in ('mode', 'grid_step', 'chunk', 'obst_chunk'):
        if hasattr(cfg, key):
            extras[key] = getattr(cfg, key)
    return base, extras


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    # NOTE: Config.replace() rebuilds a new Config from dataclass fields and does
    # not preserve previously-attached extra attrs (grid_step/chunk/mode/etc).
    # For brute-force knobs we keep overrides as extra attributes.
    if getattr(args, 'grid_step', None) is not None:
        setattr(cfg, 'grid_step', float(args.grid_step))
    if getattr(args, 'station_chunk', None) is not None:
        setattr(cfg, 'chunk', int(args.station_chunk))
    if getattr(args, 'obst_chunk', None) is not None:
        setattr(cfg, 'obst_chunk', int(args.obst_chunk))
    return cfg


def _load_gn_csv_np(cfg: Config) -> np.ndarray:
    csv_name = f'gn_coords_{cfg.num_users}.csv'
    csv_path = os.path.join(Config.training().results_dir, 'data', csv_name)
    return pd.read_csv(csv_path, header=None).values.astype(np.float32, copy=False)


def _load_gn_csv(cfg: Config) -> torch.Tensor:
    gnd_array = _load_gn_csv_np(cfg)
    return torch.tensor(gnd_array, dtype=torch.float32, device=cfg.device).reshape(-1, cfg.num_users, 3)


def _save_csv_append(path: str, filename: str, data: list, num_users: int) -> None:
    createDirectory(path)
    full = os.path.join(path, filename)
    df = pd.DataFrame(data, columns=_result_columns(num_users))
    df.to_csv(full, index=False, mode='a', header=not os.path.exists(full))
    logging.info('saved: %s', full)


def _argmax_station_for_one_gn(station_pos: torch.Tensor, gnd_nodes: torch.Tensor, obst_pts: torch.Tensor, cfg: Config):
    best_val = torch.tensor(float('-inf'), device=cfg.device, dtype=station_pos.dtype)
    best_idx = torch.tensor(0, device=cfg.device, dtype=torch.long)
    for start in range(0, station_pos.size(0), cfg.chunk):
        sig = calc_sig_strength_gpu(
            station_pos[start:start + cfg.chunk],
            gnd_nodes,
            obst_pts,
            cfg=cfg,
        )
        chunk_val, chunk_rel = torch.max(sig, dim=0)
        chunk_idx = chunk_rel.to(dtype=torch.long) + torch.tensor(start, device=cfg.device, dtype=torch.long)

        better = chunk_val > best_val
        best_val = torch.where(better, chunk_val, best_val)
        best_idx = torch.where(better, chunk_idx, best_idx)

    return int(best_idx.item()), float(best_val.item())


def _progress_tick(state: dict, inc: int = 1) -> None:
    q = state.get('queue')
    if q is None:
        return
    state['pending'] = int(state.get('pending', 0)) + int(inc)
    every = int(state.get('every', 100))
    if state['pending'] >= every:
        # ('chunk', n): heartbeat for long per-sample work
        q.put(('chunk', state['pending']))
        state['pending'] = 0


def _progress_sample_done(q) -> None:
    if q is not None:
        q.put(('sample', 1))


def run_brute_force(cfg: Config) -> None:
    set_random_seed(cfg)
    logging.info('Brute-force: users=%d height=%s grid_step=%s', cfg.num_users, cfg.height, cfg.grid_step)

    obst_list = create_obstacle_data(dot_num=0.1, cfg=cfg, return_type='list')
    gnd_nodes_all = _load_gn_csv(cfg)
    station_pos, grid_side = _build_station_grid(cfg)
    obst_pts = torch.cat(
        [torch.tensor(o.points, dtype=torch.float32, device=cfg.device) for o in obst_list],
        dim=1,
    ).T

    results = []
    for i in tqdm(range(gnd_nodes_all.size(0)), desc='Searching'):
        gnd_nodes = gnd_nodes_all[i]
        best_flat, _ = _argmax_station_for_one_gn(station_pos, gnd_nodes, obst_pts, cfg)

        row = best_flat // grid_side
        col = best_flat % grid_side
        x_max = row * cfg.grid_step - (cfg.area_size // 2)
        y_max = col * cfg.grid_step - (cfg.area_size // 2)
        z_max = cfg.height

        results.append(list(gnd_nodes.cpu().numpy().ravel()) + [x_max, y_max, z_max])

    _save_csv_append(
        path=cfg.results_dir,
        filename=f'mode_{cfg.mode}_U{cfg.num_users}_H{cfg.height}.csv',
        data=results,
        num_users=cfg.num_users,
    )


def _parse_devices(devices: str) -> list[int]:
    return [int(x) for x in devices.split(',') if x.strip()]


def _load_gn_from_shared_memory(shm_name: str, shape: tuple[int, int], start: int, end: int) -> np.ndarray:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        # IMPORTANT: Copy before closing shm; otherwise callers may hold a view
        # backed by an unmapped buffer, which can segfault.
        return np.array(arr[start:end], copy=True)
    finally:
        shm.close()


def _worker_bruteforce(
    rank: int,
    device_id: int,
    cfg_dict: dict,
    cfg_extras: dict,
    start: int,
    end: int,
    out_csv: str,
    shm_name: str,
    shm_shape: tuple[int, int],
    progress_queue,
    progress_every: int = 100,
) -> None:
    cfg = Config(**cfg_dict)
    for key, value in (cfg_extras or {}).items():
        setattr(cfg, key, value)
    cfg.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(device_id)

    set_random_seed(cfg)

    obst_list = create_obstacle_data(dot_num=0.1, cfg=cfg, return_type='list')
    station_pos, grid_side = _build_station_grid(cfg)
    obst_pts = torch.cat(
        [torch.tensor(o.points, dtype=torch.float32, device=cfg.device) for o in obst_list],
        dim=1,
    ).T

    gnd_np = _load_gn_from_shared_memory(shm_name, shm_shape, start, end)
    gnd_nodes_all = torch.tensor(gnd_np, dtype=torch.float32, device=cfg.device).reshape(-1, cfg.num_users, 3)

    results = []
    local_n = int(gnd_nodes_all.size(0))
    progress_state = {'queue': progress_queue, 'every': progress_every, 'pending': 0}
    for i in range(local_n):
        gnd_nodes = gnd_nodes_all[i]
        # Update progress per station-chunk so the global tqdm doesn't look stuck.
        for start in range(0, station_pos.size(0), cfg.chunk):
            sig = calc_sig_strength_gpu(
                station_pos[start:start + cfg.chunk],
                gnd_nodes,
                obst_pts,
                cfg=cfg,
            )

            chunk_val, chunk_rel = torch.max(sig, dim=0)
            chunk_idx = chunk_rel.to(dtype=torch.long) + torch.tensor(start, device=cfg.device, dtype=torch.long)

            if start == 0:
                best_val = chunk_val
                best_idx = chunk_idx
            else:
                better = chunk_val > best_val
                best_val = torch.where(better, chunk_val, best_val)
                best_idx = torch.where(better, chunk_idx, best_idx)

            _progress_tick(progress_state, 1)

        best_flat = int(best_idx.item())

        row = best_flat // grid_side
        col = best_flat % grid_side
        x_max = row * cfg.grid_step - (cfg.area_size // 2)
        y_max = col * cfg.grid_step - (cfg.area_size // 2)
        z_max = cfg.height

        results.append(list(gnd_nodes.cpu().numpy().ravel()) + [x_max, y_max, z_max])

        _progress_sample_done(progress_queue)

    if progress_queue is not None:
        if progress_state.get('pending', 0):
            progress_queue.put(('chunk', int(progress_state['pending'])))
        progress_queue.put(None)  # worker done

    pd.DataFrame(results, columns=_result_columns(cfg.num_users)).to_csv(out_csv, index=False)


def _terminate_worker_tree(p: mp.Process, timeout: float = 120.0) -> bool:
    """Gracefully terminate a process with timeout (no force-kill).

    Returns True if process terminated successfully within timeout.
    """
    if not p.is_alive():
        return True

    p.terminate()
    try:
        p.join(timeout=timeout)
    except Exception as e:
        logging.warning(f'join timeout for pid {p.pid}: {e}')
        return False

    if p.is_alive():
        logging.warning(f'pid {p.pid} did not terminate within {timeout}s')
        return False
    return True


def _cleanup_worker_processes(procs: list, timeout: float = 120.0) -> int:
    """Terminate all worker processes with timeout (no force-kill)."""
    not_terminated = 0
    for p in procs:
        if not _terminate_worker_tree(p, timeout=timeout):
            not_terminated += 1
    return not_terminated


def run_brute_force_multi_gpu(cfg: Config, devices: list[int]) -> None:
    set_random_seed(cfg)
    gnd_array = _load_gn_csv_np(cfg)
    total = int(gnd_array.shape[0])
    if total <= 0:
        raise ValueError('No samples found')

    shm = shared_memory.SharedMemory(create=True, size=gnd_array.nbytes)
    shm_shape = (int(gnd_array.shape[0]), int(gnd_array.shape[1]))
    procs = []
    try:
        np.ndarray(shm_shape, dtype=np.float32, buffer=shm.buf)[:] = gnd_array

        createDirectory(cfg.results_dir)
        tmp_dir = os.path.join(cfg.results_dir, '_tmp')
        createDirectory(tmp_dir)

        per = int(math.ceil(total / len(devices)))
        jobs = []
        cfg_dict, cfg_extras = _cfg_spawn_payload(cfg)
        for rank, dev in enumerate(devices):
            s = rank * per
            e = min(total, (rank + 1) * per)
            if s >= e:
                continue
            out_csv = os.path.join(tmp_dir, f'partial_rank{rank}_gpu{dev}.csv')
            jobs.append((rank, dev, cfg_dict, cfg_extras, s, e, out_csv, shm.name, shm_shape))

        ctx = mp.get_context('spawn')
        progress_queue = ctx.Queue()
        for args in jobs:
            p = ctx.Process(target=_worker_bruteforce, args=(*args, progress_queue, 10))
            p.start()
            procs.append(p)

        logging.info(f'started {len(procs)} worker(s)')

        pbar = tqdm(
            total=total,
            desc=f'Brute-force ({cfg.mode}: {cfg.num_users if cfg.mode == "num_gu" else cfg.height})',
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stderr,
        )
        done = 0
        chunk_heartbeat = 0
        last_heartbeat_t = time.time()
        max_idle_time = 300.0  # 5 minutes without heartbeat = likely hang
        last_progress_t = time.time()
        worker_error = None

        try:
            while done < len(procs):
                # Check for worker crashes.
                for i, p in enumerate(procs):
                    if p.exitcode is not None and p.exitcode != 0:
                        worker_error = f'worker {i} (pid {p.pid}) exited with code {p.exitcode}'
                        break

                if worker_error:
                    logging.error(worker_error)
                    break

                # Check for hang: no progress for too long.
                elapsed_idle = time.time() - last_progress_t
                if elapsed_idle > max_idle_time and any(p.is_alive() for p in procs):
                    worker_error = f'hung: no progress for {elapsed_idle:.1f}s'
                    logging.error(worker_error)
                    break

                try:
                    msg = progress_queue.get(timeout=1.0)
                except queue.Empty:
                    if not any(p.is_alive() for p in procs):
                        break
                    continue

                last_progress_t = time.time()

                if msg is None:
                    done += 1
                    logging.debug(f'worker sentinel received: done {done}/{len(procs)}')
                else:
                    kind, inc = msg
                    if kind == 'sample':
                        pbar.update(int(inc))
                    elif kind == 'chunk':
                        chunk_heartbeat += int(inc)
                        now = time.time()
                        dt = max(1e-6, now - last_heartbeat_t)
                        if dt >= 1.0:
                            rate = chunk_heartbeat / dt
                            pbar.set_postfix_str(f'{rate:.2f} chunks/s')
                            chunk_heartbeat = 0
                            last_heartbeat_t = now

        except KeyboardInterrupt:
            logging.warning('KeyboardInterrupt: terminating workers...')
            worker_error = 'KeyboardInterrupt'
        finally:
            pbar.close()

        # Cleanup: only terminate workers if there was an error or hang.
        if worker_error:
            logging.info('cleaning up worker processes due to error (120s timeout)...')
            not_terminated = _cleanup_worker_processes(procs, timeout=120.0)
            if not_terminated > 0:
                logging.warning(f'{not_terminated} worker(s) did not terminate within timeout')
            raise RuntimeError(f'brute-force workers failed: {worker_error}')

        # Normal completion: wait for all workers to finish gracefully.
        for i, p in enumerate(procs):
            p.join(timeout=120.0)
            if p.exitcode is not None and p.exitcode != 0:
                raise RuntimeError(f'worker {i} failed: {p.exitcode}')

        partials = sorted(glob.glob(os.path.join(tmp_dir, 'partial_rank*_gpu*.csv')))
        if not partials:
            raise RuntimeError('no partial CSV files written by workers')

        merged = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
        out_name = f'mode_{cfg.mode}_U{cfg.num_users}_H{cfg.height}.csv'
        merged.to_csv(os.path.join(cfg.results_dir, out_name), index=False)
        logging.info('merged %d partials: %s', len(partials), os.path.join(cfg.results_dir, out_name))
    finally:
        # Always cleanup workers in case of exception (120s grace period).
        if procs:
            for p in procs:
                if p.is_alive():
                    _terminate_worker_tree(p, timeout=120.0)

        shm.close()
        shm.unlink()
        logging.info('shared memory cleaned up')


if __name__ == '__main__':
    # Disable NVML in PyTorch to avoid NVML-related hangs/crashes
    os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '0')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['num_gu', 'height', 'both'], default='num_gu')
    parser.add_argument('--devices', default='', help="Comma-separated CUDA device ids, e.g. '0,1'")
    parser.add_argument('--grid-step', type=float, default=None, help='Override station grid step size')
    parser.add_argument('--station-chunk', type=int, default=None, help='Override station batch size per iteration')
    parser.add_argument('--obst-chunk', type=int, default=None, help='Override obstacle-point block size (memory/perf tradeoff)')
    args = parser.parse_args()

    devices = _parse_devices(args.devices) if args.devices else []

    if args.mode in ('num_gu', 'both'):
        for cfg_num in Config.brute_force_gen(mode='num_gu'):
            cfg_num = _apply_overrides(cfg_num, args)
            run_brute_force_multi_gpu(cfg_num, devices) if devices else run_brute_force(cfg_num)

    if args.mode in ('height', 'both'):
        for cfg_h in Config.brute_force_gen(mode='height'):
            cfg_h = _apply_overrides(cfg_h, args)
            run_brute_force_multi_gpu(cfg_h, devices) if devices else run_brute_force(cfg_h)

    logging.info('All brute-force sweeps finished.')