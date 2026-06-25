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
        # Y_df: unique_id, ds, y, where y is the (scaled) values of each series
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
    - root_path (str): Path of the local CSV file
    - data_path (str): Name of the benchmark data (e.g., ETTh1, ETTm2, etc.)
    - from_csv (bool): False, read data from NeuralForecast. True, provide a local CSV file
    - use_time_features (bool): Default is False. Enable or disable time stamp covariates
    - features (str): 'M' and 'MS' use every channel as both input and target. 'S' is for univariate
    - freq (str): 'h' or 'min' sampling frequency of the datapoints
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
        assert split in ['train', 'test', 'val'], "split must be one of ['train', 'test', 'val']"
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

        feat= self.features.lower()
        # this model produces multivariate output (it predicts ALL input channels), so 'M' and 'MS'
        # are equivalent here: every channel is used as both input and target. 'S' is univariate
        if feat in ('m', 'ms'):
            cols_data= [c for c in df_raw.columns if c != 'date']
        elif feat == 's':
            cols_data= [self.target]
        else:
            raise ValueError(f"features must be 'M', 'MS', or 'S'; got {self.features!r}")
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
        df_stamp= df_raw[['date']].iloc[border1:border2].copy()
        dt:pd.Series= pd.to_datetime(df_stamp['date'])
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

            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        else:
            return seq_x, seq_y, index


    def __len__(self):
        return len(self.inputs) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Custom(Dataset):
    """
    PyTorch Dataset for either CSVs or in-memory DataFrames from NeuralForecast, producing
    sliding windows with optional time‐feature encoding.
    - root_path (str): Path of the local CSV file
    - data_path (str): Name of the benchmark data (e.g., Weather, ECL, etc.)
    - from_csv (bool): False, read data from NeuralForecast. True, provide a local CSV file
    - use_time_features (bool): Default is False. Enable or disable time stamp covariates
    - features (str): 'M' and 'MS' use every channel as both input and target. 'S' is for univariate
    - freq (str): 'h' or 'min' sampling frequency of the datapoints
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
        assert split in ['train', 'test', 'val'], "split must be one of ['train', 'test', 'val']"
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

        feat= self.features.lower()
        # this model produces multivariate output (it predicts ALL input channels), so 'M' and 'MS'
        # are equivalent here: every channel is used as both input and target. 'S' is univariate
        if feat in ('m', 'ms'):
            cols_data= [c for c in df_raw.columns if c != 'date']
        elif feat == 's':
            cols_data= [self.target]
        else:
            raise ValueError(f"features must be 'M', 'MS', or 'S'; got {self.features!r}")
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
        df_stamp= df_raw[['date']].iloc[border1:border2].copy()
        dt:pd.Series= pd.to_datetime(df_stamp['date'])
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

            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        else:
            return seq_x, seq_y, index


    def __len__(self):
        return len(self.inputs) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_GlobalTemp(Dataset):
    """
    PyTorch Dataset for npy files of the Global Temp Dataset, producing sliding windows with
    time‐feature encoding.
    - root_path (str): Path of the local CSV file
    - data_path (str): Name of the benchmark data
    - use_time_features (bool): Default is False. Enable or disable time stamp covariates
    - freq (str): 'h' or 'min' sampling frequency of the datapoints
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
        dt:pd.Series= pd.to_datetime(df_stamp['date'])
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

            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        else:
            return seq_x, seq_y, index


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
