import torch
from torch.utils.data import DataLoader

from obstacles import create_obstacle_data
from utils.tools import calc_loss
from utils.config import Config

def train_pipeline(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, cfg: Config) -> float:

    total_loss = 0.0
    obst_tensor = create_obstacle_data(device=cfg.device, return_type='tensor')
    model.train()
    for x in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        x_reshaped = torch.tensor(cfg.scaler.inverse_transform(x.cpu()), device=cfg.device, dtype=torch.float32).view(-1, cfg.num_users, 2)
        x_reshaped = torch.cat(
            (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=cfg.device)),
            dim=-1
        )
        y_pred = torch.hstack(
            (y_pred * cfg.area_size - cfg.area_size / 2, torch.ones(y_pred.shape[0], 1, device=cfg.device) * cfg.height)
        )
        loss = calc_loss(y_pred, x_reshaped, obst_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(dataloader)


def val_pipeline(model: torch.nn.Module, dataloader: DataLoader, obst_tensor: torch.Tensor, cfg: Config) -> float:

    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            y_pred = model(x)
            x_reshaped = torch.tensor(cfg.scaler.inverse_transform(x.cpu()),
                                      device=cfg.device, dtype=torch.float32).view(-1, cfg.num_users, 2)
            x_reshaped = torch.cat(
                (x_reshaped, torch.zeros((x_reshaped.shape[0], x_reshaped.shape[1], 1), device=cfg.device)),
                dim=-1
            )
            y_pred = torch.hstack(
                (y_pred* cfg.area_size - cfg.area_size / 2, torch.ones(y_pred.shape[0], 1, device=cfg.device) * cfg.height)
            )
            total_loss += calc_loss(y_pred, x_reshaped, obst_tensor).item()

    return total_loss/len(dataloader)
