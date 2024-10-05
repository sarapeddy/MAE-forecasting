# MAE: Implementation with 2 - step pipeline

<p align="center">
    <img src="/images/mae.png" alt="mae" width=600>
</p>

## Dependencies
A list of dependencies is provided in ```requirements.txt```. After creating a virtual environment, we recommend installing dependencies via ```pip```:

```shell
pip install -r /path/to/requirements.txt
```
## Run the experiments

To run the experiments on forecasting task use the following command:

```shell
sh forecasting.sh
```
After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/forecasting/B{batch_size}_E{output_repr_dim}/<mode>/DatasetName__RunName_Date_Time/`.
