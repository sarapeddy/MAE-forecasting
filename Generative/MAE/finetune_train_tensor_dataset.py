import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from moving_avg_tensor_dataset import MovingAvg

class TimeSeriesDatasetWithMovingAvg_Finetune(TensorDataset):

    def __init__(self, original_dataset: Tensor, n_time_cols,  kernel_size=9, seq_len=336, pred_len=24, labelled_ratio=0.1, mode='train', dataset_name='ETTh1'):
        self.n_time_cols = n_time_cols
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = MovingAvg(kernel_size, stride=1)
        x_time = original_dataset[:, :, :self.n_time_cols]
        self.x_original = original_dataset[:, :, self.n_time_cols:]
        x_avg = self.moving_avg(self.x_original)
        x_err = self.x_original - x_avg

        self.x_time = x_time[:, :-self.pred_len, :].unfold(dimension=1, size=self.seq_len, step=1)
        self.x_avg = x_avg[:, :-self.pred_len, :].unfold(dimension=1, size=self.seq_len, step=1)
        self.x_err = x_err[:, :-self.pred_len, :].unfold(dimension=1, size=self.seq_len, step=1)
        self.data_y = self.x_original[:, self.seq_len:, :].unfold(dimension=1, size=self.pred_len, step=1)
        self.x_time = torch.swapaxes(self.x_time, 0,1)
        self.x_avg = torch.swapaxes(self.x_avg, 0,1)
        self.x_err = torch.swapaxes(self.x_err, 0,1)
        self.data_y = torch.swapaxes(self.data_y, 0,1)
        expanded_dataset = torch.cat([self.x_time, self.x_avg, self.x_err], dim=2)

        if self.mode == 'train' and dataset_name!='national_illness':
            n_samples = int(labelled_ratio * expanded_dataset.shape[0])
            indices = torch.randperm(expanded_dataset.shape[0])[:n_samples]
            expanded_dataset = expanded_dataset[indices]
            self.data_y = self.data_y[indices]

        super(TimeSeriesDatasetWithMovingAvg_Finetune, self).__init__(expanded_dataset)

    def __getitem__(self, index):

        data_y = self.data_y[index]
        el = self.tensors[0][index]
        n_feature_cols = (el.shape[1] - self.n_time_cols) // 2
        time_data = el[:, :self.n_time_cols, :]
        avg_data = el[:, self.n_time_cols:self.n_time_cols + n_feature_cols, :]
        err_data = el[:, -n_feature_cols:, :]

        x_avg = torch.cat([time_data, avg_data], dim=1)
        x_err = torch.cat([time_data, err_data], dim=1)
        return x_avg, x_err, data_y

    def __len__(self):
        return len(self.tensors[0])



class TimeSeriesDataset_Finetune(TensorDataset):
    def __init__(self, original_dataset: Tensor, n_time_cols,  kernel_size=9, seq_len=336, pred_len=24, labelled_ratio=0.1, mode='train', dataset_name='ETTh1'):
        self.n_time_cols = n_time_cols
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.x_original = original_dataset[:, :, self.n_time_cols:]

        self.original_dataset = original_dataset[:, :-self.pred_len, :].unfold(dimension=1, size=self.seq_len, step=1)
        self.data_y = self.x_original[:, self.seq_len:, :].unfold(dimension=1, size=self.pred_len, step=1)

        self.original_dataset = torch.swapaxes(self.original_dataset, 0,1)
        self.data_y = torch.swapaxes(self.data_y, 0,1)

        if self.mode == 'train' and dataset_name!='national_illness':
            n_samples = int(labelled_ratio * self.original_dataset.shape[0])
            indices = torch.randperm(self.original_dataset.shape[0])[:n_samples]
            self.original_dataset = self.original_dataset[indices]
            self.data_y = self.data_y[indices]

        super(TimeSeriesDataset_Finetune, self).__init__(self.original_dataset)

    def __getitem__(self, index):

        data_y = self.data_y[index]
        el = self.tensors[0][index]

        return el, data_y

    def __len__(self):
        return len(self.tensors[0])