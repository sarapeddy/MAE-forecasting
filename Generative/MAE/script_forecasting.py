import argparse
import math

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import *
import torch.utils.data as Data
from build_dataset import CustomTensorDataset
from utils import yaml_config_hook
from datautils import load_forecast_csv
from save_model import save_model
from moving_avg_tensor_dataset import TimeSeriesDatasetWithMovingAvg
from model_dlinear import MAE_ViT_Dlinear
import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def custom_collate_fn(batch, n_time_cols=7):
    # Stack della lista di tensori in un unico tensore
    data = torch.stack([item[0] for item in batch], dim=0)
    total_covariate = (data.shape[2] - n_time_cols)//2

    result_data_avg = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols:n_time_cols+total_covariate]], dim=2)
    result_data_err = torch.cat([data[:, :, :n_time_cols], data[:, :, n_time_cols + total_covariate:]], dim=2)
    return result_data_avg, result_data_err

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MAE")
    # parser.add_argument("--dataset", default='ETTh1', type=str)
    # args = parser.parse_args()

    config = yaml_config_hook(f"config/ETTh1_config_MAE.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps'
    print("Device:", device)

    # Load dataset
    print("Dataset:", args.dataset)

    print("-------------- LOAD DATASET: PREPROCESSING ------------------------")

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = load_forecast_csv(args.dataset)
    dataset_name = args.dataset

    train_data = data[:, train_slice]
    vali_data = data[:, valid_slice]
    test_data = data[:, test_slice]

    if args.mode == 'dlinear':
        train_dataset = TimeSeriesDatasetWithMovingAvg(torch.from_numpy(train_data).to(torch.float), n_time_cols=n_time_cols, pred_len=args.pred_len)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=load_batch_size,
            shuffle=True,
            # collate_fn=custom_collate_fn
        )

        model = MAE_ViT_Dlinear(
            sample_shape=[args.n_channel + n_time_cols, args.n_length],
            patch_size=(args.n_channel + n_time_cols, args.patch_size),
            mask_ratio=args.mask_ratio
        ).to(device)
    else:
        train_dataset = CustomTensorDataset(
            data=(data[:, train_slice])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=load_batch_size,
            shuffle=True,
        )

        model = MAE_ViT(
            sample_shape=[args.n_channel, args.n_length],
            patch_size=(args.n_channel, 16),
            mask_ratio=args.mask_ratio
        ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_learning_rate * args.batch_size / 256,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    min_loss = 100
    for e in range(args.total_epoch):
        model.train()
        print('===== start training ======')
        losses = []
        # for sample, label in tqdm(iter(train_loader)):
        for sample_avg, sample_err, y in tqdm(iter(train_loader)):
            step_count += 1

            sample_avg = sample_avg.swapaxes(1, 2)
            sample_avg = np.expand_dims(sample_avg, axis=1)

            sample_err = sample_err.swapaxes(1, 2)
            sample_err = np.expand_dims(sample_err, axis=1)

            sample_avg = torch.tensor(sample_avg, dtype=torch.float32).to(device)
            sample_err = torch.tensor(sample_err, dtype=torch.float32).to(device)



            predicted_sample, mask = model(sample_avg, sample_err)
            loss = torch.mean((predicted_sample - (sample_avg + sample_err)) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' save pre-trained model '''
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_model(args, model, optim)
            print("Model update with loss {}.".format(min_loss))

    print("Finished")