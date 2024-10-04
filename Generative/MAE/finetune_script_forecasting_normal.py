import json
import os
import argparse
import math

import torch.nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random
from model import *
from utils import yaml_config_hook
from datautils import load_forecast_csv
from finetune_train_tensor_dataset import TimeSeriesDataset_Finetune

from model import MAE_ViT, ViT_Forecasting

import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_avg(result_list):
    return sum(result_list)/len(result_list)

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MAE")
    parser.add_argument('--dataset', default='electricity', type=str)
    parser.add_argument('--mode', default='MAE', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_device_batch_size', default=64, type=int)
    parser.add_argument('--base_learning_rate', default=1.5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--total_epoch', default=200, type=int)
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--model_path', default='training', type=str)
    parser.add_argument('--n_length', default=336, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--labelled_ratio', default=0.1, type=float)
    parser.add_argument('--finetune_batch_size', default=64, type=int)
    parser.add_argument('--finetune_base_learning_rate', default=0.001, type=float)
    parser.add_argument('--finetune_epochs', default=31, type=int)
    parser.add_argument('--finetune_seed', default=42, type=int)
    parser.add_argument('--pretrain', default=True, type=bool)
    parser.add_argument('--run_name', default='forecast_multivar',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')

    args = parser.parse_args()

    setup_seed(args.finetune_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)

    # Load dataset
    print("Dataset:", args.dataset)

    print("-------------- LOAD DATASET: PREPROCESSING ------------------------")

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_time_cols = load_forecast_csv(args.dataset)
    dataset_name = args.dataset
    print("Dataset:", args.dataset)

    train_data = data[:, train_slice]
    vali_data = data[:, valid_slice]
    test_data = data[:, test_slice]

    ours_result = {}
    loss_train_avg = {}
    loss_val_avg = {}
    loss_test_avg = {}

    for pred_len in pred_lens:
        train_dataset = TimeSeriesDataset_Finetune(
            original_dataset=torch.from_numpy(train_data).to(torch.float),
            n_time_cols=n_time_cols,
            seq_len=args.n_length,
            pred_len=pred_len,
            labelled_ratio=args.labelled_ratio,
            mode='train',
            dataset_name=dataset_name
        )

        # load val data
        val_dataset = TimeSeriesDataset_Finetune(
            original_dataset=torch.from_numpy(vali_data).to(torch.float),
            n_time_cols=n_time_cols,
            seq_len=args.n_length,
            pred_len=pred_len,
            labelled_ratio=args.labelled_ratio,
            mode='val',
            dataset_name=dataset_name
        )

        # load test data
        test_dataset = TimeSeriesDataset_Finetune(
            original_dataset=torch.from_numpy(test_data).to(torch.float),
            n_time_cols=n_time_cols,
            seq_len=args.n_length,
            pred_len=pred_len,
            labelled_ratio=args.labelled_ratio,
            mode='test',
            dataset_name=dataset_name
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            args.finetune_batch_size,
            shuffle=True,
            drop_last=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            args.finetune_batch_size,
            shuffle=True,
            drop_last=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            args.finetune_batch_size,
            shuffle=True,
            drop_last=True
        )

        model_enc_dec = MAE_ViT(
            sample_shape=[train_data.shape[0]*train_data.shape[2], args.n_length],
            patch_size=(train_data.shape[0]*train_data.shape[2], args.patch_size),
            mask_ratio=args.mask_ratio
        )

        global arch
        if args.pretrain == True:
            model_fp = os.path.join(args.model_path, "forecasting/B{}_E{}/{}/Pretrained_{}_{}_{}".format(
                args.batch_size, args.emb_dim, args.mode, args.dataset, args.emb_dim, pred_len),
                                    "Pretrained_{}_{}_{}".format(args.dataset, args.emb_dim, pred_len) + ".pkl")
            model_enc_dec.load_state_dict(torch.load(model_fp, map_location=args.device.type))

            arch = args.dataset
            print("With pretrain.")
        else:
            arch = args.dataset + "_no_pre"
            print("No pretrain.")

        # model = ViT_Forecasting(model.encoder, n_covariate=args.n_channel, pred_len=args.pred_len).to(args.device)


        model = ViT_Forecasting(model_enc_dec.encoder, n_covariate=train_data.shape[-1] - n_time_cols, pred_len=pred_len, n_sample=train_data.shape[0]).to(args.device)

        Finetune_mode = "Full"  # or "Partial"
        if Finetune_mode == "Full":
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.head.parameters(),
                                          lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                          betas=(0.9, 0.999), weight_decay=args.weight_decay)

        criterion = torch.nn.MSELoss()

        loss_fn = torch.nn.MSELoss()

        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

        best_val_mse = 0
        step_count = 0
        optimizer.zero_grad()

        for epoch in range(args.finetune_epochs):
            print("--------start fine-tuning--------")
            model.train()
            loss_epoch = []
            mse_epoch = []
            mae_epoch = []

            for sample, y in tqdm(iter(train_dataloader)):
                step_count += 1
                sample = sample.to(args.device)
                sample = sample.reshape(sample.shape[0], 1, sample.shape[1] * sample.shape[2], sample.shape[3])

                y = y.to(args.device)
                logits = model(sample)

                y = y.reshape(y.shape[0], y.shape[1]*y.shape[2]*y.shape[3])
                loss = loss_fn(logits, y)

                metrics = cal_metrics(logits.detach().cpu(), y.detach().cpu())


                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                mae_epoch.append(metrics['MAE'])
                mse_epoch.append(metrics['MSE'])


            lr_scheduler.step()
            avg_train_loss = sum(loss_epoch) / len(loss_epoch)
            avg_mse = sum(mse_epoch) / len(mse_epoch)
            avg_mae = sum(mae_epoch) / len(mae_epoch)

            if pred_len in loss_train_avg:
                loss_train_avg[pred_len].append(avg_train_loss)
            else:
                loss_train_avg[pred_len]=[avg_train_loss]

            if epoch % 10 == 0:
                print("Epoch [{}/{}]\n average Finetune Loss: {}\n MAE: {}\n MSE: {}\n".format(epoch, args.finetune_epochs, avg_train_loss, avg_mae, avg_mse))


            model.eval()

            print("--------start validation--------")
            with torch.no_grad():
                loss_epoch = []
                mse_epoch = []
                mae_epoch = []

                for sample, y in tqdm(iter(val_dataloader)):

                    sample = sample.to(args.device)
                    sample = sample.reshape(sample.shape[0], 1, sample.shape[1] * sample.shape[2],
                                                    sample.shape[3])
                    y = y.to(args.device)
                    logits = model(sample)

                    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2]*y.shape[3])
                    loss = loss_fn(logits, y)

                    metrics = cal_metrics(logits.detach().cpu(), y.detach().cpu())

                    loss_epoch.append(loss.item())
                    mae_epoch.append(metrics['MAE'])
                    mse_epoch.append(metrics['MSE'])

                avg_val_loss = get_avg(loss_epoch)
                avg_val_mae = get_avg(mae_epoch)
                avg_val_mse = get_avg(mse_epoch)
                if pred_len in loss_val_avg:
                    loss_val_avg[pred_len].append(avg_val_loss)
                else:
                    loss_val_avg[pred_len] = [avg_val_loss]


                if epoch % 10 == 0:
                    print("Epoch [{}/{}]\n average Finetune val Loss: {}\n Avarage MAE: {}\n Avarage MSE: {}\n".format(epoch,
                                                                                                   args.finetune_epochs,
                                                                                                   avg_val_loss, avg_val_mae,
                                                                                                   avg_val_mse))


            # use F1 to select best model
            if avg_val_mse < best_val_mse or epoch == 0:
                best_val_mse = avg_val_mse
                print(f'saving best model with F1 {best_val_mse} at {epoch} epoch!')
                if not os.path.exists(f'save/finetune/{args.mode}/{arch}'):
                    os.makedirs(f'save/finetune/{args.mode}/{arch}')
                FT_model_path = f'save/finetune/{args.mode}/{arch}/' + f'{arch}_{pred_len}' + str(args.labelled_ratio) + '.pkl'
                torch.save(model, FT_model_path)

            # if epoch % 10 == 0:

            print("TEST-epoch {}--------start testing-------- ".format(epoch))

            model.eval()
            with torch.no_grad():
                loss_epoch = []
                mse_epoch = []
                mae_epoch = []

                mse_epoch_inv = []
                mae_epoch_inv = []

                for sample_avg, y in tqdm(iter(test_dataloader)):
                    sample = sample.to(args.device)
                    sample = sample.reshape(sample.shape[0], 1, sample.shape[1] * sample.shape[2], sample.shape[3])

                    y = y.to(args.device)
                    logits = model(sample)

                    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2]*y.shape[3])
                    loss = loss_fn(logits, y)

                    metrics = cal_metrics(logits.detach().cpu(), y.detach().cpu())

                    logits = logits.reshape(logits.shape[0], pred_len, -1)
                    y = y.reshape(y.shape[0], pred_len, -1)
                    logits = logits.reshape(logits.shape[0]*logits.shape[1], -1)
                    y = y.reshape(y.shape[0]*y.shape[1], -1)

                    metrics_inverse = cal_metrics(scaler.inverse_transform(logits.detach().cpu()), scaler.inverse_transform(y.detach().cpu()))

                    loss_epoch.append(loss.item())
                    mae_epoch.append(metrics['MAE'].item())
                    mse_epoch.append(metrics['MSE'].item())
                    mse_epoch_inv.append(metrics_inverse['MSE'].item())
                    mae_epoch_inv.append(metrics_inverse['MAE'].item())

                avg_test_loss = get_avg(loss_epoch)
                avg_test_mse = get_avg(mse_epoch)
                avg_test_mae = get_avg(mae_epoch)
                avg_test_mse_inv = get_avg(mse_epoch_inv)
                avg_test_mae_inv = get_avg(mae_epoch_inv)
                if pred_len in loss_test_avg:
                    loss_test_avg[pred_len].append(avg_test_loss)
                else:
                    loss_test_avg[pred_len] = [avg_test_loss]


                print("Testing: \n Average test Loss: {}\n Test MAE: {}\n Test MSE: {}\n".format(avg_test_loss, avg_test_mae, avg_test_mse))
                print("Pretrain: {}; Label ratio: {}".format(args.pretrain, args.labelled_ratio))

                if avg_val_mse == best_val_mse:
                    ours_result[int(pred_len)] = {
                        'norm': {
                            'MAE': avg_test_mae,
                            'MSE': avg_test_mse
                        },
                        'raw': {
                            'MAE': avg_test_mae_inv,
                            'MSE': avg_test_mse_inv
                        }
                    }

    with open(f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{args.mode}/{dir}/eval_res.json',
              'w') as f:
        eval_res = {
            'ours': ours_result
        }
        json.dump(eval_res, f, indent=4)

    with open(
            f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{args.mode}/{dir}/loss_train_avg.json',
            'w') as f:
        json.dump(loss_train_avg, f, indent=4)

    with open(f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{args.mode}/{dir}/loss_val_avg.json',
              'w') as f:
        json.dump(loss_val_avg, f, indent=4)

    with open(
            f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{args.mode}/{dir}/loss_test_avg.json',
            'w') as f:
        json.dump(loss_test_avg, f, indent=4)
