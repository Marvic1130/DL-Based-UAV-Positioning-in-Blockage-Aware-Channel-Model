import os
import pandas as pd
import torch
from tqdm import trange
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb

from datasets import TrainDataset, BlockageDataset
from obstacles import create_obstacle_data
from model import Net
from train import train_pipeline, val_pipeline
from utils.config import Config, set_random_seed
from utils.tools import createDirectory

if __name__ == '__main__':
    cfg = Config.training()
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

        wandb.init(project="DL-based UAV Positioning", name=f"num_gu_training : {test_cfg.num_users}",
                   config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        best_loss = float('inf')

        for epoch in trange(test_cfg.epochs, desc=f"Training with num of gu={test_cfg.num_users}"):

            train_loss = train_pipeline(model, train_dataloader, optimizer, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.num_users]["train_loss"].append(train_loss)
            results[test_cfg.num_users]["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                            'models','num_gu',
                                                            f'best_num_gu_{test_cfg.num_users}.pt'))


            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                    'models','num_gu',
                                                    f'gn_num_{test_cfg.num_users}_epoch_{test_cfg.epochs - 1}.pt'))
        wandb.finish()

    result_list = []
    for num_gu, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]),
                                                                   start=1):
            result_list.append(
                {"num_gu": num_gu, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
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

        wandb.init(project="DL-based UAV Positioning", name=f"height_training: {test_cfg.height}",
                   config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        best_loss = float('inf')

        for epoch in trange(test_cfg.epochs, desc=f"Training with height={test_cfg.height}"):

            train_loss = train_pipeline(model, train_dataloader, optimizer, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.height]["train_loss"].append(train_loss)
            results[test_cfg.height]["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                                            'models','height',
                                                            f'best_height_{test_cfg.height}.pt'))

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        torch.save(model.state_dict(), os.path.join(test_cfg.results_dir,
                                 'models','height',
                                 f'height_{test_cfg.height}_epoch_{test_cfg.epochs - 1}.pt'))
        wandb.finish()

    result_list = []
    for height, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]),
                                                                   start=1):
            result_list.append(
                {"height": height, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    df_results = pd.DataFrame(result_list)
    createDirectory(cfg.results_dir)
    df_results.to_csv(os.path.join(cfg.results_dir, 'height_result.csv'), index=False)
    print("Train complete for height.")
    