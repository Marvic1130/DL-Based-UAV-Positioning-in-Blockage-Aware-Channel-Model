import torch
from torch.utils.data import DataLoader

from obstacles import create_obstacle_data
from utils.tools import calc_loss
from utils.config import Config

def train_pipeline(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   cfg: Config) -> float:
    """
    Train the model for one epoch using the provided training data and configuration settings.

    This function performs a single training epoch on the given model. For each batch from the dataloader,
    it performs the following steps:
      - Zeroes out the gradients.
      - Feeds the input data through the model to obtain predictions.
      - Uses the scaler in the configuration to inverse transform the input data and reshapes it to
        a tensor of shape (batch_size, num_users, 2). A zero-filled z-coordinate is concatenated to
        form 3D coordinates.
      - Scales and shifts the model output using cfg.area_size and cfg.height to match the original
        coordinate range.
      - Computes the loss using the calc_loss function with the processed predictions, reshaped inputs,
        and obstacle tensor.
      - Performs backpropagation and updates the model parameters.
      - Accumulates the loss over all batches.

    :param model: The neural network model to be trained.
    :param dataloader: DataLoader providing training batches.
    :param optimizer: Optimizer used to update the model parameters.
    :param cfg: A Config instance containing training and environment settings.
    :return: The average training loss per batch over the epoch.
    """
    total_loss = 0.0
    obst_tensor = create_obstacle_data(cfg=cfg, return_type='tensor')
    model.train()
    for x in dataloader:
        optimizer.zero_grad()
        y_pred = model(x)
        x_reshaped = torch.tensor(cfg.scaler.inverse_transform(x.cpu()), device=cfg.device, dtype=torch.float32).view(
            -1, cfg.num_users, 2)
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

    return total_loss / len(dataloader)


def val_pipeline(model: torch.nn.Module, dataloader: DataLoader, obst_tensor: torch.Tensor, cfg: Config) -> float:
    """
    Evaluate the model on the validation dataset using the provided configuration settings.

    This function sets the model to evaluation mode and processes the validation data without
    computing gradients. For each validation batch, it:
      - Passes the input through the model to get predictions.
      - Inverse transforms the input data using the scaler from the configuration and reshapes it
        to a tensor of shape (batch_size, num_users, 2), then concatenates a zero-filled z-coordinate.
      - Scales and shifts the model predictions using cfg.area_size and cfg.height.
      - Computes the loss using calc_loss with the processed predictions, reshaped inputs, and obstacle tensor.
      - Accumulates the loss over all batches.

    :param model: The neural network model to evaluate.
    :param dataloader: DataLoader providing validation batches.
    :param obst_tensor: A torch.Tensor containing the obstacle point data.
    :param cfg: A Config instance containing training and environment settings.
    :return: The average validation loss per batch.
    """
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
                (y_pred * cfg.area_size - cfg.area_size / 2,
                 torch.ones(y_pred.shape[0], 1, device=cfg.device) * cfg.height)
            )
            total_loss += calc_loss(y_pred, x_reshaped, obst_tensor).item()

    return total_loss / len(dataloader)
