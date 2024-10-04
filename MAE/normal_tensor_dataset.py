from torch import Tensor
from torch.utils.data import TensorDataset


class TimeSeriesDataset(TensorDataset):

    def __init__(self, original_dataset: Tensor, seq_len=336, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        if original_dataset.shape[0] == 1:
            expanded_dataset = original_dataset.squeeze(0)
        else:
            expanded_dataset = original_dataset.swapaxes(0, 1)
            expanded_dataset = expanded_dataset.reshape(expanded_dataset.shape[0], expanded_dataset.shape[1]*expanded_dataset.shape[2])
        super(TimeSeriesDataset, self).__init__(expanded_dataset)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        data_x = self.tensors[0][s_begin:s_end]
        data_y = self.tensors[0][r_begin:r_end]
        return data_x, data_y

    def __len__(self):
        return len(self.tensors[0]) - self.seq_len - self.pred_len + 1