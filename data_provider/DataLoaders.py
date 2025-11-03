# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Data Loaders and Processing Methods
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from datasetsforecast.long_horizon import LongHorizon
from .TimeFeature import time_features



def load_data(name, unique_id=None):
    """
    Load ETT, Weather, or Traffic datasets via neuralforecast's LongHorizon, pivot it to wide format,
    optionally sub-select unique_id series, and return the combined DataFrame plus metadata.
    """
    name= name.lower()
    # name in {'ettm1','ettm2','etth1','etth2'}
    if name == 'ettm1':
        # ETT data contains several different transformer units; each gets its own unique_id
        # Y_df: unique_id, ds, y, where y is the (scaled) values of each serie
        # X_df: unique_id, ds, ex_1, ..., ex_4, where these are calendar (exogenous) covariates
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='ETTm1')
    elif name == 'ettm2':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='ETTm2')
    elif name == 'etth1':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='ETTh1')
    elif name == 'etth2':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='ETTh2')
    elif name == 'weather':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='Weather')
    elif name == 'traffic':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='TrafficL')
    elif name == 'ecl':
        Y_df, X_df, *_= LongHorizon.load(directory='./', group='ECL')
    else:
        raise ValueError(f"Unknown dataset {name}")

    # pivot Y_df to wide format: index=ds, columns=unique_id, values=y
    df= (
        Y_df
        .assign(ds= lambda d: pd.to_datetime(d['ds']))  # ensure datetime in 'ds'
        .pivot(index='ds', columns='unique_id', values='y')
        .rename_axis(columns=None)                      # drop the name 'unique_id'
        .reset_index()
        .rename(columns={'ds': 'date'})
    )

    # filter down to only the requested series (if any)
    if unique_id is not None:
        missing= set(unique_id) - set(df.columns)
        if missing:
            raise KeyError(f"Requested series {missing} not found in '{name}'")
        # always include OT as key target
        unique_id= set(unique_id).union({'OT'})
        cols= ['date', *unique_id]
        df= df[cols]

    target= 'OT'  # oil temperature
    features= [c for c in df.columns if c not in ('date', target)]

    if name in ("ettm1", "ettm2"):
        val_size = 11520
        test_size= 11520
        freq= 'min'
    elif name in ("etth1", "etth2"):
        val_size = 2880
        test_size= 2880
        freq= 'h'
    elif name == 'weather':
        val_size = 5270
        test_size= 10539
        freq= 'min'
    elif name == 'traffic':
        val_size = 1756
        test_size= 3508
        freq= 'h'
    else:  # name == 'ecl'
        val_size = 2632
        test_size= 5260
        freq= 'h'

    return df, features, target, val_size, test_size, freq



class Dataset_ETT(Dataset):
    """
    PyTorch Dataset for either the original ETT CSVs or in-memory DataFrames from NeuralForecast,
    producing sliding windows with optional time‐feature encoding.
    The original ETT‐style loader returns:
    - inputs: history data for training (seq_len)
    - targets: history tail (label_len) U forecast horizon (pred_len)
    - data_stamp: calendar covariates for inputs and targets
    This formulation is useful for training a model that "sees" the tail of the history and then
    predicts the next pred_len steps in one shot.
    - We differ from the original class and return all data with permuted orders, i.e.,
    (batch_size, channels/features, seq_len).
    """

    def __init__(self, root_path, data_path, from_csv=True, split='train', size=None,
                 features='MS', target='OT', scale=True, timeenc=1, use_time_features=False):
        assert data_path.lower() in ['ettm1','ettm2','etth1','etth2'], \
                "data_path should be 'ETTm1', 'ETTm2', 'ETTh1', or 'ETTh2'"
        assert split in ['train', 'test', 'val']
        # init
        type_map= {'train': 0, 'val': 1, 'test': 2}
        self.set_type= type_map[split]
        self.from_csv= from_csv
        self.features= features
        self.target= target
        self.scale= scale
        self.timeenc= timeenc
        self.use_time_features= use_time_features

        # size [seq_len, label_len, pred_len]
        # info -- default window sizes: 384 / 96 / 96
        if size is None:
            self.seq_len  = 24 * 4 * 4
            self.label_len= 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len  = size[0]  # look‑back window: how many past steps you feed into the model
            self.label_len= size[1]  # input length: how many of the last look‑back steps you "re‑see"
            self.pred_len = size[2]  # forecast horizon: how many future steps you predict

        # 'T' or 'min' is the consistent naming for minutes
        self.freq= 'h' if 'etth' in data_path.lower() else 'min'

        if self.from_csv:
            # here we will get data from a csv file
            data_path= data_path + '.csv'
        # otherwise, we will read a DataFrame coming from NeuralForecast

        self.root_path= root_path
        self.data_path= data_path
        self.__read_data__()


    def __read_data__(self):
        if self.from_csv:
            df_raw= pd.read_csv(os.path.join(str(self.root_path), str(self.data_path)))
        else:
            df_raw, *_= load_data(self.data_path)

        multiple= 1 if self.freq== 'h' else 4
        months  = 12 * 30 * 24 * multiple

        border1s= [0, months - self.seq_len, months + 4 * 30 * 24 * multiple - self.seq_len]
        border2s= [months, months + 4 * 30 * 24 * multiple, months + 8 * 30 * 24 * multiple]
        border1= border1s[self.set_type]
        border2= border2s[self.set_type]

        if self.features.lower()== 'm' or self.features.lower()== 'ms':
            cols_data= [c for c in df_raw.columns if c != 'date']
        else:  # elif self.features.lower()== 's':
            cols_data= [self.target]
        df_data= df_raw[cols_data]

        # scaling
        self.scaler= StandardScaler()
        if self.scale:
            # fit only on the training slice
            train_data= df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data= self.scaler.transform(df_data.values)
        else:
            data= df_data.values

        # build time‐stamp features
        df_stamp= df_raw[['date']][border1:border2]
        dt= pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month']  = dt.dt.month
            df_stamp['day']    = dt.dt.day
            df_stamp['weekday']= dt.dt.weekday
            df_stamp['hour']   = dt.dt.hour
            if self.freq== 'min':
                df_stamp['minute']= (dt.dt.minute // 15)

            data_stamp= df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # pass the DataFrame slice directly
            data_stamp= time_features(df_stamp, freq=self.freq)
            data_stamp= data_stamp.transpose(1, 0)
        else:
            raise ValueError("timeenc must be 0 or 1")

        # inputs shape:  (T, dim)
        self.inputs = data[border1:border2]
        # targets shape: (H, dim)
        self.targets= data[border1:border2]
        # calendar covariates
        self.data_stamp= data_stamp


    def __getitem__(self, index):
        s_begin= index
        s_end  = s_begin + self.seq_len
        r_begin= s_end   - self.label_len
        r_end  = r_begin + self.label_len + self.pred_len

        seq_x= torch.from_numpy(self.inputs[s_begin:s_end]).permute(1, 0).float()
        seq_y= torch.from_numpy(self.targets[r_begin:r_end]).permute(1, 0).float()

        if self.use_time_features:
            seq_x_mark= torch.from_numpy(self.data_stamp[s_begin:s_end]).permute(1, 0).float()
            seq_y_mark= torch.from_numpy(self.data_stamp[r_begin:r_end]).permute(1, 0).float()

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            return seq_x, seq_y


    def __len__(self):
        return len(self.inputs) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Custom(Dataset):
    """
    PyTorch Dataset for either CSVs or in-memory DataFrames from NeuralForecast, producing
    sliding windows with optional time‐feature encoding.
    - freq: 'h' or 'min'
    Returns:
    - inputs: history data for training (seq_len)
    - targets: history tail (label_len) U forecast horizon (pred_len)
    - data_stamp: calendar covariates for inputs and targets
    This formulation is useful for training a model that "sees" the tail of the history and then
    predicts the next pred_len steps in one shot.
    - We differ from the original class and return all data with permuted orders, i.e.,
    (batch_size, channels/features, seq_len).
    """

    def __init__(self, root_path, data_path, from_csv=True, split='train', size=None, features='MS',
                 target='OT', scale=True, timeenc=1, freq='h', use_time_features=False):
        assert split in ['train', 'test', 'val']
        # init
        type_map= {'train': 0, 'val': 1, 'test': 2}
        self.set_type= type_map[split]
        self.from_csv= from_csv
        self.features= features
        self.target= target
        self.scale= scale
        self.timeenc= timeenc
        # 'T' or 'min' is the consistent naming for minutes
        self.freq= freq
        self.use_time_features= use_time_features

        # size [seq_len, label_len, pred_len]
        # info -- default window sizes: 384 / 96 / 96
        if size is None:
            self.seq_len  = 24 * 4 * 4
            self.label_len= 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len  = size[0]  # look‑back window: how many past steps you feed into the model
            self.label_len= size[1]  # input length: how many of the last look‑back steps you "re‑see"
            self.pred_len = size[2]  # forecast horizon: how many future steps you predict

        if self.from_csv:
            # here we will get data from a csv file
            data_path= data_path + '.csv'
        # otherwise, we will read a DataFrame coming from NeuralForecast

        self.root_path= root_path
        self.data_path= data_path
        self.__read_data__()


    def __read_data__(self):
        if self.from_csv:
            df_raw= pd.read_csv(os.path.join(str(self.root_path), str(self.data_path)))
            train_size= int(len(df_raw) * 0.7)
            test_size = int(len(df_raw) * 0.2)
            val_size  = len(df_raw) - train_size - test_size
        else:
            df_raw, _, target, val_size, test_size, freq= load_data(self.data_path)
            self.target= target
            self.freq= freq
            train_size= len(df_raw) - val_size - test_size

        border1s= [0, train_size - self.seq_len, len(df_raw) - test_size - self.seq_len]
        border2s= [train_size, train_size + val_size, len(df_raw)]
        border1= border1s[self.set_type]
        border2= border2s[self.set_type]

        if self.features.lower()== 'm' or self.features.lower()== 'ms':
            cols_data= [c for c in df_raw.columns if c != 'date']
        else:  # elif self.features.lower()== 's':
            cols_data= [self.target]
        df_data= df_raw[cols_data]

        # scaling
        self.scaler= StandardScaler()
        if self.scale:
            # fit only on the training slice
            train_data= df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data= self.scaler.transform(df_data.values)
        else:
            data= df_data.values

        # build time‐stamp features
        df_stamp= df_raw[['date']][border1:border2]
        dt= pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month']  = dt.dt.month
            df_stamp['day']    = dt.dt.day
            df_stamp['weekday']= dt.dt.weekday
            df_stamp['hour']   = dt.dt.hour
            if self.freq== 'min':
                df_stamp['minute']= (dt.dt.minute // 15)

            data_stamp= df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # pass the DataFrame slice directly
            data_stamp= time_features(df_stamp, freq=self.freq)
            data_stamp= data_stamp.transpose(1, 0)
        else:
            raise ValueError("timeenc must be 0 or 1")

        # inputs shape:  (T, dim)
        self.inputs = data[border1:border2]
        # targets shape: (H, dim)
        self.targets= data[border1:border2]
        # calendar covariates
        self.data_stamp= data_stamp


    def __getitem__(self, index):
        s_begin= index
        s_end  = s_begin + self.seq_len
        r_begin= s_end   - self.label_len
        r_end  = r_begin + self.label_len + self.pred_len

        seq_x= torch.from_numpy(self.inputs[s_begin:s_end]).permute(1, 0).float()
        seq_y= torch.from_numpy(self.targets[r_begin:r_end]).permute(1, 0).float()

        if self.use_time_features:
            seq_x_mark= torch.from_numpy(self.data_stamp[s_begin:s_end]).permute(1, 0).float()
            seq_y_mark= torch.from_numpy(self.data_stamp[r_begin:r_end]).permute(1, 0).float()

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            return seq_x, seq_y


    def __len__(self):
        return len(self.inputs) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_GlobalTemp(Dataset):
    """
    PyTorch Dataset for npy files of the Global Temp Dataset, producing sliding windows with
    time‐feature encoding.
    - freq: 'h' or 'min'
    Returns:
    - inputs: history data for training (seq_len)
    - targets: history tail (label_len) U forecast horizon (pred_len)
    - data_stamp: calendar covariates for inputs and targets
    Adapted from https://doi.org/10.1038/s42256-023-00667-9
    """

    def __init__(self, root_path, data_path='temp_global_hourly_', time_path='data_time_', split='train',
                 size=None, features='S', target=0, scale=True, train_scaler=None, timeenc=1, freq='h',
                 use_time_features=True, data_cleaner=None, verbose=True):
        assert split in ['train', 'test', 'val'], "split must be one of ['train', 'test', 'val']"
        # init
        self.set_type= split
        self.features= features
        self.target= target
        self.scale= scale
        self.scaler= train_scaler
        self.timeenc= timeenc
        self.data_cleaner= data_cleaner
        self.chosen_idx= None
        # 'T' or 'min' is the consistent naming for minutes
        self.freq= freq
        self.use_time_features= use_time_features
        self.verbose= verbose

        # size [seq_len, label_len, pred_len]
        # info -- default window sizes: 384 / 96 / 96
        if size is None:
            self.seq_len  = 24 * 4 * 4
            self.label_len= 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len  = size[0]  # look‑back window: how many past steps you feed into the model
            self.label_len= size[1]  # input length: how many of the last look‑back steps you "re‑see"
            self.pred_len = size[2]  # forecast horizon: how many future steps you predict

        self.root_path= root_path
        self.data_path= data_path
        self.time_path= time_path
        self.__read_data__()


    def __read_data__(self):
        self.raw_data= np.load(
            os.path.join(self.root_path, self.data_path + self.set_type + ".npy"), allow_pickle=True
        )  # expected shape (T, stations, feats) -> (17519, 34040, 3)
        self.raw_time= np.load(
            os.path.join(self.root_path, self.time_path + self.set_type + ".npy"), allow_pickle=True
        )  # expected shape (T,) -> (17519)
        if self.verbose:
            print(self.raw_data.shape)
            print("==== " + self.set_type + " data sorted load finished ====")

        # select feature subset
        raw_data= self.raw_data
        raw_time= self.raw_time
        if self.features == 'S':
            # keep first feature dimension per station -> shape (T, stations, 1)
            raw_data= raw_data[:, :, :1]
        elif self.features == 'S_station':
            # require integer station target index
            if not isinstance(self.target, (int, np.integer)):
                raise ValueError("For features='S_station' 'target' must be an integer index.")
            raw_data= raw_data[:, self.target:(self.target + 1), :1]
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        # ---------------- data cleaning -----------------
        T, S, C= raw_data.shape

        if self.data_cleaner is not None and self.set_type == 'train':
            # compute selection on train split only
            data, chosen_idx= self.data_cleaner.quality_score_per_channel(raw_data)
            self.chosen_idx = np.asarray(chosen_idx, dtype=int)
        elif self.data_cleaner is not None:
            # for val/test: require that the same DataCleaning instance was used for train
            if self.data_cleaner.chosen_idx is None:
                raise RuntimeError("data_cleaner.chosen_idx not set: construct train dataset first or provide chosen_idx.")

            chosen_idx= np.asarray(self.data_cleaner.chosen_idx, dtype=int)

            if chosen_idx.max() >= S:
                raise IndexError("chosen_idx contains indices >= number of stations in current file.")
            data= raw_data[:, chosen_idx, :].astype(np.float64, copy=False)
            self.chosen_idx= chosen_idx
        else:
            data= raw_data.astype(np.float64, copy=False)

        if self.data_cleaner is not None and self.verbose:
            print(data.shape)
            print("==== " + self.set_type + " data cleaning finished ====")
        # ---------------- data cleaning -----------------

        # reshape into (T, stations * feat)
        data_len, station, feat= data.shape
        data= data.reshape(data_len, station * feat)  # (17519, 34040*3)

        # --- scaling (z-score per column across time) ---
        if self.scale and self.set_type == 'train':
            self.scaler= StandardScaler()
            # fit the scaler only on the training slice
            self.scaler.fit(data)

        if self.scale:
            assert self.scaler is not None, "train_scaler cannot be None"
            # apply the scaler that was fitted on the training slice
            data= self.scaler.transform(data)
        # ------------------- scaling --------------------

        # build time‐stamp features
        df_stamp= pd.DataFrame(data=raw_time, columns=['date'])
        dt= pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            # explicit integer fields per timestamp -> produce shape (T, n_time_feats)
            df_stamp['month']  = dt.dt.month
            df_stamp['day']    = dt.dt.day
            df_stamp['weekday']= dt.dt.weekday
            df_stamp['hour']   = dt.dt.hour
            if self.freq== 'min':
                df_stamp['minute']= (dt.dt.minute // 15)

            data_stamp= df_stamp.drop(['date'], axis=1).values  # shape (T, n_feats)
        elif self.timeenc == 1:
            # pass the DataFrame slice directly -- expected (n_feats, T)
            data_stamp= time_features(df_stamp, freq=self.freq)
            data_stamp= data_stamp.transpose(1, 0)
        else:
            raise ValueError("timeenc must be 0 or 1")

        self.data_x= data
        self.data_stamp= data_stamp


    def __getitem__(self, index):
        s_begin= index
        s_end  = s_begin + self.seq_len
        r_begin= s_end - self.label_len
        r_end  = r_begin + self.label_len + self.pred_len

        seq_x= torch.from_numpy(self.data_x[s_begin:s_end]).permute(1, 0).float()
        seq_y= torch.from_numpy(self.data_x[r_begin:r_end]).permute(1, 0).float()

        if self.use_time_features:
            seq_x_mark= torch.from_numpy(self.data_stamp[s_begin:s_end]).permute(1, 0).float()
            seq_y_mark= torch.from_numpy(self.data_stamp[r_begin:r_end]).permute(1, 0).float()

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            return seq_x, seq_y


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1



class SlidingWindowMultivariate(Dataset):
    """
    Given a (T x D) pandas DataFrame, build multivariate sliding-window supervised examples.
    Splits the last 'val_size' windows for validation, last 'test_size' for testing,
    and uses the rest for training.
    """

    def __init__(self, data, input_width, history_tail, output_width, covariate_cols, target_col,
                 val_size, test_size, split="train", scale=True, scaler=None) -> None:
        assert split in ("train", "val", "test")
        self.split= split
        self.input_width = input_width
        self.history_tail= history_tail
        self.output_width= output_width

        all_cols= covariate_cols + [target_col]
        arr= data[all_cols].values.astype(np.float32)  # (T, dim)
        T, dim= arr.shape

        # how many siding window
        window   = input_width + output_width
        n_windows= T - window + 1
        # cutoff between train / eval in terms of windows
        train_end= n_windows - (val_size + test_size)
        val_end  = n_windows - test_size

        # fit / apply the scaler before windowing
        if scale and split.lower()== 'train':
            # fit only on all time‐steps that appear in any train window
            # the raw training portion extends through the last window's end
            train_slice= arr[: train_end + window]
            self.scaler= StandardScaler()
            self.scaler.fit(train_slice)
            # transform the entire array so that both train/val/test are on the same scale
            arr= self.scaler.transform(arr)
        elif scale:
            # val / test must be given an existing scaler
            assert scaler is not None, "Scaler must be provided for val/test splits"
            self.scaler= scaler
            # transform the entire array so that both train/val/test are on the same scale
            arr= self.scaler.transform(arr)
        else:
            self.scaler= None

        # build the raw sliding‐window view
        X= np.lib.stride_tricks.sliding_window_view(arr, window_shape=(window, dim))
        # here X has shape (n_windows, 1, window, dim)
        X= X.squeeze(1)
        # now X has shape  (n_windows, window, dim)

        # split into X_all and Y_all
        X_all= X[:, :input_width, :]  # (n_windows, input_width, dim)
        Y_all= X[:, input_width:, :]  # (n_windows, output_width, dim)

        # select only the windows belonging to the requested split
        if split.lower()== 'train':
            idx= slice(0, train_end)
        elif split.lower()== 'val':
            idx= slice(train_end, val_end)
        else:  # test
            idx= slice(val_end, n_windows)

        # convert to tensors from (B, T, dim) format to (B, dim, T)
        # inputs shape:  (B, T, dim) -> permute -> (B, dim, T)
        self.inputs = (torch.from_numpy(X_all[idx].copy())).permute(0, 2, 1).float()
        # targets shape: (B, H, dim) -> permute -> (B, dim, H)
        self.targets= (torch.from_numpy(Y_all[idx].copy())).permute(0, 2, 1).float()
        self.length= self.inputs.shape[0]


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        """
        RETURNS: A tuple with data and its targets.
        """
        x, y_future= self.inputs[idx], self.targets[idx]

        if self.history_tail:
            # grab the "history tail" of x along time:
            y_history= x[:, -self.history_tail:]
            y_full= torch.cat([y_history, y_future], dim=1)

            return x, y_full

        return x, y_future


    def inverse_transform(self, arr:np.ndarray) -> np.ndarray:
        """
        Invert standardization.
        - 'arr' should be shape (n_samples, n_features).
        """
        assert self.scaler is not None, "Scaler was not provided"
        return self.scaler.inverse_transform(arr)



class SlidingWindowDataset(Dataset):
    """
    Builds sliding-window supervised examples (X, y) from a wide-format ETT dataframe.
    Splits the last 'val_size' windows for validation, last 'test_size' for testing,
    and uses the rest for training.
    """

    def __init__(self, data, input_width, history_tail, output_width, covariate_cols, target_col,
                 val_size, test_size, split="train", scale=True, scaler=None) -> None:
        assert split in ("train", "val", "test")
        self.split= split
        self.input_width = input_width
        self.history_tail= history_tail
        self.output_width= output_width

        size= len(data)
        # fit / apply the scaler before windowing
        train_end= size - (val_size + test_size)
        df_train = data.iloc[:train_end].copy()
        df_eval  = data.iloc[train_end:].copy()

        all_cols= covariate_cols + [target_col]
        if scale and split.lower()== 'train':
            # fit a new scaler on training data only
            self.scaler= StandardScaler()
            arr_train= df_train[all_cols].to_numpy()
            self.scaler.fit(arr_train)
            # transform df_train in-place
            df_train.loc[:, all_cols]= self.scaler.transform(arr_train)
        elif scale:
            # for val / test must be given a pre-fitted scaler
            assert scaler is not None, "Scaler must be provided for val/test"
            self.scaler= scaler

            df_val = df_eval.iloc[:val_size].copy()
            df_test= df_eval.iloc[val_size:].copy()

            if split.lower()== 'val':
                arr_val= df_val[all_cols].to_numpy()
                # transform df_val in-place
                df_val.loc[:, all_cols]= self.scaler.transform(arr_val)
            else:  # split.lower()== 'test'
                arr_test= df_test[all_cols].to_numpy()
                # transform df_test in-place
                df_test.loc[:, all_cols]= self.scaler.transform(arr_test)

            # re‐assemble the eval scaled data
            df_eval= pd.concat([df_val, df_test], ignore_index=True)
        else:
            self.scaler= None

        # re‐assemble the full scaled data for windowing:
        data_full= pd.concat([df_train, df_eval], ignore_index=True)
        # build sliding windows after scaling
        X_windows= []
        Y_windows= []

        # (1) define X and y
        if covariate_cols:
            covs= data_full[covariate_cols].to_numpy(dtype=np.float32)
        else:
            # when handling a single-dimensional dataset
            covs= data_full[[target_col]].to_numpy(dtype=np.float32)
        target= data_full[[target_col]].to_numpy(dtype=np.float32)

        # build siding window
        window= input_width + output_width
        last_start= size - window + 1
        for t0 in range(0, last_start):
            y_end= t0 + window - 1

            # decide whether this window belongs to train/val/test
            if split.lower()== 'train':
                if y_end >= size - val_size - test_size:
                    continue
            elif split.lower()== 'val':
                if not (size - val_size - test_size <= y_end < size - test_size):
                    continue
            else:  # test
                if y_end < size - test_size:
                    continue

            x= covs[t0 : t0+input_width]
            # split history vs future
            y_hist  = target[t0+input_width- history_tail : t0+input_width].squeeze(-1)
            y_future= target[t0+input_width               : t0+input_width+output_width].squeeze(-1)

            if history_tail:
                y= np.concatenate([y_hist, y_future])
            else:
                y= y_future

            X_windows.append(x)
            Y_windows.append(y)

        # (2) stack into numpy arrays -- more efficient than directly convert to tensor
        X_np= np.stack(X_windows, axis=0)
        Y_np= np.stack(Y_windows, axis=0)

        # (3) convert to tensors and permute inputs to (B, channels/features, seq_length)
        self.inputs= (torch.from_numpy(X_np)).permute(0, 2, 1)
        self.targets= torch.from_numpy(Y_np)
        self.length= self.inputs.size(0)


    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        """
        RETURNS: A tuple with data and its targets.
        """
        return self.inputs[idx], self.targets[idx]


    def inverse_transform(self, arr:np.ndarray) -> np.ndarray:
        """
        Invert standardization.
        - 'arr' should be shape (n_samples, n_features).
        """
        assert self.scaler is not None, "Scaler was not provided"
        return self.scaler.inverse_transform(arr)
