import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CubeObstacle, CylinderObstacle, BlockageDataset
from utils.config import Hyperparameters as hp
from utils.tools import calc_sig_strength_gpu

logging.basicConfig(level=logging.WARNING)

def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_df(path: str, name: str, data: list):
    if os.path.exists(f"{path}/{name}"):
        df = pd.read_csv(f"{path}/{name}")
        df = pd.concat([df, pd.DataFrame(data, columns=["gnd1_x", "gnd1_y", "gnd1_z", "gnd2_x", "gnd2_y", "gnd2_z",
                                                        "gnd3_x", "gnd3_y", "gnd3_z", "gnd4_x", "gnd4_y", "gnd4_z",
                                                        "result_x", "result_y", "result_z"])])
    else:
        df = pd.DataFrame(data, columns=["gnd1_x", "gnd1_y", "gnd1_z", "gnd2_x", "gnd2_y", "gnd2_z",
                                     "gnd3_x", "gnd3_y", "gnd3_z", "gnd4_x", "gnd4_y", "gnd4_z",
                                     "result_x", "result_y", "result_z"])
    createDirectory(path)
    df.to_csv(f"{path}/{name}", index=False)

if __name__ == "__main__":
    result_ls = []

    # Create obstacles and convert to torch tensors
    obstacle_ls = [
        CubeObstacle(-30, 15, 35, 60, 20, 0.2),
        CubeObstacle(-30, -25, 45, 10, 35, 0.2),
        CylinderObstacle(0, -30, 70, 10, 0.2)
    ]

    # Create dataset
    dataset = BlockageDataset(2**20, obstacle_ls, 4, dtype=torch.float32).to(hp.device)
    logging.info(f"len(dataset): {len(dataset)}")
    dataloader = DataLoader(dataset)
    try:
        for i, data in enumerate(tqdm(dataloader)):
            station_pos, gnd_nodes, obst_points = data
            station_pos = station_pos.squeeze()
            gnd_nodes = gnd_nodes.squeeze()
            obst_points = obst_points.squeeze()

            sig = calc_sig_strength_gpu(station_pos, gnd_nodes, obst_points)
            sig = sig.reshape(hp.area_size, hp.area_size)

            max_idx = torch.unravel_index(torch.argmax(sig), sig.shape)

            np_gnd_nodes = gnd_nodes.cpu().numpy()
            np_max_pos = (max_idx[0].cpu().numpy(), max_idx[1].cpu().numpy(), 70)
            logging.info(f"Max Signal: {sig[max_idx]}, Index: {max_idx}")

            result_ls.append([np_gnd_nodes[0, 0], np_gnd_nodes[0, 1], np_gnd_nodes[0, 2], np_gnd_nodes[1, 0],
                              np_gnd_nodes[1, 1], np_gnd_nodes[1, 2], np_gnd_nodes[2, 0], np_gnd_nodes[2, 1],
                              np_gnd_nodes[2, 2], np_gnd_nodes[3, 0], np_gnd_nodes[3, 1], np_gnd_nodes[3, 2],
                              np_max_pos[0] - (hp.area_size // 2), np_max_pos[1] - (hp.area_size // 2), np_max_pos[2]])
            logging.info(f"Result: {max_idx}, sig_max: {sig[max_idx]}")

    except KeyboardInterrupt as e:
        logging.warning("Interrupted by user" + str(e))
    finally:
        save_df("data", "data.csv", result_ls)