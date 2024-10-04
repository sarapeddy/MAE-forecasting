import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TimeSeriesDatasetWithMovingAvg(TensorDataset):
    
    def __init__(self, original_dataset: Tensor, n_time_cols,  kernel_size=9, seq_len=336, pred_len=24):
        self.n_time_cols = n_time_cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = MovingAvg(kernel_size, stride=1)
        x_time = original_dataset[:, :, :self.n_time_cols]
        self.x_original = original_dataset[:, :, self.n_time_cols:]
        x_avg = self.moving_avg(self.x_original)
        x_err = self.x_original - x_avg
        expanded_dataset = torch.cat([x_time, x_avg, x_err], dim=2)
        super(TimeSeriesDatasetWithMovingAvg, self).__init__(expanded_dataset)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        data_x = self.tensors[0][:, s_begin:s_end, :]
        data_y = self.x_original[:, r_begin:r_end, :]
        time_data = data_x[:, :, :self.n_time_cols]
        n_feature_cols = (data_x.shape[2] - self.n_time_cols)//2
        avg_data = data_x[:, :, self.n_time_cols:self.n_time_cols+n_feature_cols]
        err_data = data_x[:, :, -n_feature_cols:]
        x_avg = torch.cat([time_data, avg_data], dim=2)
        x_err = torch.cat([time_data, err_data], dim=2)
        x_avg = x_avg.swapaxes(0, 1)
        x_err = x_err.swapaxes(0, 1)
        data_y = data_y.swapaxes(0, 1)
        x_avg = x_avg.reshape(x_avg.shape[0], x_avg.shape[1]*x_avg.shape[2])
        x_err = x_err.reshape(x_err.shape[0], x_err.shape[1]*x_err.shape[2])
        data_y = data_y.reshape(data_y.shape[0], data_y.shape[1]*data_y.shape[2])
        return x_avg, x_err, data_y

    def __len__(self):
        return self.tensors[0].shape[1] - self.seq_len - self.pred_len + 1
