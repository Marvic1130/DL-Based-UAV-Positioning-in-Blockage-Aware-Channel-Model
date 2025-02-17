import pandas as pd
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wandb

from datasets import CubeObstacle, CylinderObstacle, TrainDataset
from model import Net
from utils.tools import calc_loss
from utils.config import Hyperparameters as hp

random_seed = 42
batch_size = 1024

obstacle_ls = [
    CubeObstacle(-30, 25, 35, 60, 20, 0.1),
    CubeObstacle(-30, -25, 45, 10, 35, 0.1),
    CubeObstacle(-30, -60, 35, 60, 20, 0.1),
    CubeObstacle(50, -20, 35, 25, 25, 0.1),
    CylinderObstacle(10, -5,  70, 15, 0.1),
]

if __name__ == '__main__':

    obst_points = []
    for obstacle in obstacle_ls:
        obst_points.append(torch.tensor(obstacle.points, dtype=torch.float32))

    obst_points = torch.cat([op for op in obst_points], dim=1).mT.to(hp.device)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    x = pd.read_csv('./data/dataset.csv')
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    x_scaled = scaler_x.fit_transform(x)

    x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=random_seed)

    train_dataset = TrainDataset(x_train, dtype=torch.float32).to(hp.device)
    val_dataset = TrainDataset(x_val, dtype=torch.float32).to(hp.device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr_ls = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    results = {lr: {"train_loss": [], "val_loss": []} for lr in lr_ls}

    for lr in lr_ls:

        wandb.init(project="DL-based UAV Positioning", name=f"lr_test: {lr}", config={
            "batch_size": batch_size,
            "epochs": 1000,
            "random_seed": random_seed,
            "learning_rates": lr
        })

        # 모델 및 옵티마이저 초기화
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(hp.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for epoch in trange(1000, desc=f"Training with lr={lr}"):
            train_loss = 0.0
            model.train()
            for x in train_dataloader:
                optimizer.zero_grad()
                y_pred = model(x)

                # x_reshaped 생성
                x_reshaped = torch.tensor(scaler_x.inverse_transform(x.cpu()), device=hp.device,
                                          dtype=torch.float32).view(-1, 4, 2)
                x_reshaped = torch.cat(
                    (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=hp.device)), dim=-1)

                # y_pred 수정 및 손실 계산
                y_pred = torch.hstack((y_pred, torch.ones(y_pred.shape[0], 1, device=hp.device) * 0.7)) * 100
                loss = calc_loss(y_pred, x_reshaped, obst_points)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 검증 손실 계산
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for x in val_dataloader:
                    y_pred = model(x)
                    x_reshaped = torch.tensor(scaler_x.inverse_transform(x.cpu()), device=hp.device,
                                              dtype=torch.float32).view(-1, 4, 2)
                    x_reshaped = torch.cat(
                        (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=hp.device)),
                        dim=-1)
                    y_pred = torch.hstack((y_pred, torch.ones(y_pred.shape[0], 1, device=hp.device) * 0.7)) * 100
                    val_loss += calc_loss(y_pred, x_reshaped, obst_points).item()

            # 에폭별 평균 손실 기록
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            results[lr]["train_loss"].append(train_loss)
            results[lr]["val_loss"].append(val_loss)

            # wandb에 손실 로깅
            wandb.log({
                f"train_loss": train_loss,
                f"val_loss": val_loss,
                "epoch": epoch + 1
            })
        wandb.finish()
