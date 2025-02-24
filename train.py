import os
import logging
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wandb

from datasets import CubeObstacle, CylinderObstacle, TrainDataset
from model import Net
from utils.tools import calc_loss
from utils.config import Hyperparameters as hp


def train_pipeline(model, dataloader, optimizer, scaler, obst_points, device):
    total_loss = 0.0
    model.train()
    for x in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        y_pred = model(x)
        # 원본 데이터 복원 후 (4, 2) 형태로 재구성
        x_reshaped = torch.tensor(scaler.inverse_transform(x.cpu()), device=device, dtype=torch.float32).view(-1, 4, 2)
        # z 좌표(높이)는 0으로 채움
        x_reshaped = torch.cat(
            (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=device)),
            dim=-1
        )
        # y_pred에 고정 값 0.7을 추가하고 스케일 변환
        y_pred = torch.hstack(
            (y_pred, torch.ones(y_pred.shape[0], 1, device=device) * 0.7)
        ) * 100
        loss = calc_loss(y_pred, x_reshaped, obst_points)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def val_pipeline(model, dataloader, scaler, obst_points, device, **kwargs):
    """
    Validation pipeline 함수.

    Args:
        model (torch.nn.Module): 평가할 모델.
        dataloader (DataLoader): 검증 데이터셋의 DataLoader.
        scaler: 입력 데이터를 원래 스케일로 복원하기 위한 스케일러.
        obst_points (torch.Tensor): 장애물 점들.
        device (torch.device): 사용 장치.
        **kwargs: 추가 인자.
            - mode: 'visual' 일 경우 시각화 활성화.
            - num_epoch: 시각화를 몇 에폭마다 수행할지 정하는 값.
            - current_epoch: 현재 에폭 번호.

    Returns:
        float: 총 검증 손실.
    """
    total_loss = 0.0
    gn_num = kwargs.get('gn_num', 4)
    visual = kwargs.get('visual', False)
    current_epoch = kwargs.get('current_epoch', None)
    obstacle_ls = kwargs.get('obstacle_ls', None)

    # mode가 'visual'이고, current_epoch와 num_epoch가 주어졌으며, current_epoch가 num_epoch의 배수면 시각화 수행
    if visual:
        if obstacle_ls is None:
            logging.warning("obstacle_ls is not given. Skip visualization.")
            visual = False
        if current_epoch is None:
            logging.warning("current_epoch is not given. Skip visualization.")
            visual = False

    # 시각화를 위해 예측 및 정답 데이터를 저장할 리스트 (필요할 경우)
    preds = []
    gn_coords = []

    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader, desc="Validation"):
            y_pred = model(x)
            x_reshaped = torch.tensor(scaler.inverse_transform(x.cpu()),
                                      device=device, dtype=torch.float32).view(-1, gn_num, 2)
            x_reshaped = torch.cat(
                (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=device)),
                dim=-1
            )
            # y_pred에 고정값 0.7을 추가한 후 스케일 변환
            y_pred = torch.hstack(
                (y_pred, torch.ones(y_pred.shape[0], 1, device=device) * 0.7)
            ) * 100

            total_loss += calc_loss(y_pred, x_reshaped, obst_points).item()

            if visual:
                preds.append(y_pred.cpu().numpy())
                gn_coords.append(x_reshaped.cpu().numpy())

    if visual and len(preds) > 0:
        line = [go.Scatter3d(
            x=[gn_coords[0][0, i, 0], preds[0][0][0]],
            y=[gn_coords[0][0, i, 1], preds[0][0][1]],
            z=[0, 70],
            mode='lines',
            line=dict(color='green', width=5),
        ) for i in range(gn_num)]

        pred_scatter = go.Scatter3d(
            x=[preds[0][0][0]],
            y=[preds[0][0][1]],
            z=[70],
            mode='markers',
            marker=dict(color='red', size=3),
            name='Prediction'
        )
        line.append(pred_scatter)

        gn_scatter_ls = [go.Scatter3d(
            x=gn_coords[0][0, :, 0],
            y=gn_coords[0][0, :, 1],
            z=[0] * gn_num,
            mode='markers',
            marker=dict(color='blue', size=3),
            name='Ground Nodes'
        )]

        obstacle_traces = [obstacle_ls[i].plotly_obj() for i in range(len(obstacle_ls))]

        fig = go.Figure(data= line + obstacle_traces + gn_scatter_ls)
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-100, 100], title='X axis'),
                yaxis=dict(range=[-100, 100], title='Y axis'),
                zaxis=dict(range=[0, 80], title='Z axis'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            title=f"Epoch {current_epoch} Validation"
        )
        wandb.log({"Validation": fig})
        fig.show()

    return total_loss

random_seed = 42
batch_size = 1024
epochs = 10000
lr = 5e-5

save_dir = "./models/train_model"
os.makedirs(save_dir, exist_ok=True)

obstacle_ls = [
    CubeObstacle(-30, 25, 35, 60, 20, 0.1),
    CubeObstacle(-30, -25, 45, 10, 35, 0.1),
    CubeObstacle(-30, -60, 35, 60, 20, 0.1),
    CubeObstacle(50, -20, 35, 25, 25, 0.1),
    CylinderObstacle(10, -5,  70, 15, 0.1),
]

if __name__ == '__main__':

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if hp.device == "cuda":
        torch.cuda.manual_seed_all(random_seed)

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

    model = Net(x_train.shape[1], 1024, 4, output_N=2).to(hp.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    wandb.init(project="DL-based UAV Positioning", name=f"train model", config={
        "batch_size": batch_size,
        "epochs": epochs,
        "random_seed": random_seed,
        "learning_rate": lr
    })

    for epoch in range(epochs):

        model.train()
        train_loss = train_pipeline(model, train_dataloader, optimizer, scaler_x, obst_points, hp.device)
        visual = False
        if epoch % 500 == 0 or epoch == epochs-1: visual=True
        val_loss = val_pipeline(model, val_dataloader, scaler_x, obst_points, hp.device, visual=visual, current_epoch=epoch, obstacle_ls=obstacle_ls)
        # 에폭별 평균 손실 기록
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        if epoch % 500 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at: {checkpoint_path}")

        # wandb에 손실 로깅
        wandb.log({
            f"train_loss": train_loss,
            f"val_loss": val_loss,
            "epoch": epoch + 1
        })
    wandb.finish()
