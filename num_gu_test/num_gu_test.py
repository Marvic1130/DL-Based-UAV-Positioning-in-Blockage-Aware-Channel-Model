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
    cfg = Config.gu_num_test()
    results = {num_gu: {"train_loss": [], "val_loss": []} for num_gu in cfg.test_list}

    for test_cfg in Config.gu_num_test_gen():
        set_random_seed(test_cfg)
        obstacle_ls, obst_tensor = create_obstacle_data(cfg=test_cfg, return_type='both')

        x = BlockageDataset(test_cfg.num_samples, obstacle_ls=obstacle_ls, cfg=test_cfg).gnd_nodes[:, :, :2].reshape(-1,
                                                                                                           test_cfg.num_users * 2).cpu()
        x_scaled = test_cfg.scaler.transform(x)

        x_train, x_val = train_test_split(x_scaled, test_size=0.2, random_state=test_cfg.random_seed)

        train_dataset = TrainDataset(x_train, dtype=torch.float32).to(test_cfg.device)
        val_dataset = TrainDataset(x_val, dtype=torch.float32).to(test_cfg.device)

        train_dataloader = DataLoader(train_dataset, batch_size=test_cfg.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=test_cfg.batch_size, shuffle=False)

        wandb.init(project="DL-based UAV Positioning", name=f"num_gu_test: {test_cfg.lr}", config=test_cfg.to_dict())

        set_random_seed(test_cfg)
        model = Net(train_dataset.x.shape[1], 1024, 4, output_N=2).to(test_cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=test_cfg.lr)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        for epoch in trange(test_cfg.epochs, desc=f"Training with num of gu={test_cfg.num_users}"):
            train_loss = train_pipeline(model, train_dataloader, optimizer, test_cfg)
            val_loss = val_pipeline(model, val_dataloader, obst_tensor, test_cfg)

            results[test_cfg.num_users]["train_loss"].append(train_loss)
            results[test_cfg.num_users]["val_loss"].append(val_loss)

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })
        wandb.finish()

    result_list = []
    for num_gu, res in results.items():
        for epoch, (train_loss, val_loss) in enumerate(zip(res["train_loss"], res["val_loss"]), start=1):
            result_list.append({"num_gu": num_gu, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    df_results = pd.DataFrame(result_list)
    createDirectory(cfg.results_dir)
    df_results.to_csv(os.path.join(cfg.results_dir, 'result.csv'), index=False)
    print("Training complete.")
    print("Results:")
    print(df_results)