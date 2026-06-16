# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Get DataLoader objects for ETT and other datasets.
"""

import numpy as np
from torch.utils.data import DataLoader
from collections.abc import Sequence
from .DataLoaders import Dataset_ETT, Dataset_Custom, Dataset_GlobalTemp



def loaders_safety_checks(patch_width:int, block_size:int, width_factor:float, test_horizons:int|Sequence[int]|None):
    # --- safety checks ---
    if patch_width <= 0 or block_size <= 0:
        raise ValueError("patch_width and block_size must be positive.")
    if patch_width > block_size:
        raise ValueError(f"patch_width ({patch_width}) cannot exceed block_size ({block_size}).")
    if width_factor < 0.0:
        raise ValueError("width_factor must be a non-negative number.")
    if int(patch_width * width_factor) > block_size:
        raise ValueError("patch_width * width_factor cannot exceed block_size (would make label_len negative).")

    if isinstance(test_horizons, (int, np.integer)):
        test_horizons= [int(test_horizons)]
    if test_horizons is None:
        test_horizons= [96, 192, 336, 720]

    for h in test_horizons:
        if h is None:
            raise ValueError("test_horizons cannot contain None; use [] or None for 'no test'.")
        h_int= int(h)
        if h_int <= 0:
            raise ValueError(f"Invalid horizon {h}; all horizons must be positive integers.")

    return test_horizons



def get_train_val_test_sizes(patch_width:int, block_size:int, width_factor:float, test_horizons:Sequence[int],
                             is_encoder_model:bool):

    INPUT_WIDTH = block_size  # how many past steps you feed into the model

    if is_encoder_model:
        """ ----- setup for getting training and val data to feed Encoders ----- """
        # int(patch_width * width_factor) == 0, generate train/val for SSL mode (data == target)
        HISTORY_TAIL= block_size - int(patch_width * width_factor)
        OUTPUT_WIDTH= int(patch_width * width_factor)  # predict a sequence of time-patches
        size_tv= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]
    else:
        """ ----- setup for getting training and val data to feed Decoders ----- """
        HISTORY_TAIL= block_size - patch_width
        OUTPUT_WIDTH= patch_width                   # predict the next time-patch
        size_tv= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]

    sizes_te= []
    for horizon in test_horizons:
        if int(patch_width * width_factor) > 0:
            INPUT_WIDTH = block_size
            HISTORY_TAIL= 0
            OUTPUT_WIDTH= horizon
        else:  # int(patch_width * width_factor) == 0, generate test for SSL mode (data == target)
            INPUT_WIDTH = horizon
            HISTORY_TAIL= horizon
            OUTPUT_WIDTH= 0

        size_te= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]
        sizes_te.append(size_te)

    return size_tv, sizes_te



"""
ETT DataLoaders
"""


def get_ett_data_loaders(ett_root_path, dataset_name_1, dataset_name_2, from_csv, btc_size, time_covariates,
                         patch_width, block_size, width_factor, test_horizons=None, scale=True, is_encoder_model=True):
    """
    Create DataLoader objects for ETTx1, ETTx2, and combined datasets for encoders/decoders training
    and testing.
    Args:
    - ett_root_path (str): Directory path for dataset files.
    - dataset_name_1, dataset_name_2 (str): Filenames (e.g., 'ETTh1.csv', 'ETTh2.csv').
    - from_csv (bool): Whether to read data from CSV files or neuralforecast's LongHorizon.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - width_factor (float): Number of output patches for encoders.
    Returns:
    - Tuple of DataLoader and scaler objects for train and validate encoders/decoders, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, width_factor, test_horizons)
    size_tv, sizes_te= get_train_val_test_sizes(patch_width, block_size, width_factor, test_horizons, is_encoder_model)

    """ ETTx1 """
    train_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='train', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
    )
    val_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='val', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
    )
    tds_scaler_1= train_ds_ett_1.scaler

    """ ETTx2 """
    train_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='train', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
    )
    val_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='val', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
    )
    tds_scaler_2= train_ds_ett_2.scaler

    """ ETTx1 DataLoaders """
    train_loader_ett_1= DataLoader(train_ds_ett_1, batch_size=btc_size, shuffle=True)
    val_loader_ett_1  = DataLoader(val_ds_ett_1,   batch_size=btc_size, shuffle=False)

    """ ETTx2 DataLoaders """
    train_loader_ett_2= DataLoader(train_ds_ett_2, batch_size=btc_size, shuffle=True)
    val_loader_ett_2  = DataLoader(val_ds_ett_2,   batch_size=btc_size, shuffle=False)


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader_ett_1= {}
    test_loader_ett_2= {}

    for idx, horizon in enumerate(test_horizons):
        size_te= sizes_te[idx]

        test_ds_ett_1= Dataset_ETT(
            root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
        )
        test_loader_ett_1[horizon]= DataLoader(test_ds_ett_1,  batch_size=btc_size, shuffle=False)

        test_ds_ett_2= Dataset_ETT(
            root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=scale, timeenc=1, use_time_features=time_covariates
        )
        test_loader_ett_2[horizon]= DataLoader(test_ds_ett_2, batch_size=btc_size, shuffle=False)


    return (
        # Train / Val
        train_loader_ett_1, val_loader_ett_1, train_loader_ett_2, val_loader_ett_2,
        tds_scaler_1, tds_scaler_2,
        # Test -- Decoders/Encoders
        *[test_loader_ett_1[int(h)] for h in test_horizons],
        *[test_loader_ett_2[int(h)] for h in test_horizons],
    )



"""
Custom DataLoaders
"""


def get_custom_data_loaders(root_path, dataset_name, from_csv, btc_size, time_covariates, patch_width, block_size,
                            width_factor, freq='min', test_horizons=None, scale=True, is_encoder_model=True):
    """
    Create DataLoader objects for encoders/decoders training and testing.
    Args:
    - root_path (str): Directory path for dataset files.
    - dataset_name (str): Filenames (e.g., 'ETTh1', 'ETTh2', 'Weather').
    - from_csv (bool): Whether to read data from CSV files or neuralforecast's LongHorizon.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - width_factor (float): Number of output patches for encoders.
    Returns:
    - Tuple of DataLoader and scaler objects for train and validate encoders/decoders, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, width_factor, test_horizons)
    size_tv, sizes_te= get_train_val_test_sizes(patch_width, block_size, width_factor, test_horizons, is_encoder_model)

    train_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='train', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    val_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='val', size=size_tv,
        features='MS', target='OT', scale=scale, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    tds_scaler= train_ds.scaler

    """ DataLoaders """
    train_loader= DataLoader(train_ds, batch_size=btc_size, shuffle=True)
    val_loader  = DataLoader(val_ds,   batch_size=btc_size, shuffle=False)


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader= {}

    for idx, horizon in enumerate(test_horizons):
        size_te= sizes_te[idx]

        test_ds= Dataset_Custom(
            root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=scale, timeenc=1, freq=freq, use_time_features=time_covariates
        )
        test_loader[horizon]= DataLoader(test_ds,  batch_size=btc_size, shuffle=False)

    return (
        # Train / Val
        train_loader, val_loader, tds_scaler,
        # Test -- Decoders/Encoders
        *[test_loader[int(h)] for h in test_horizons],
    )



"""
Global Temp DataLoaders
"""


def get_global_temp_data_loaders(root_path='./global_temp', data_path='temp_global_hourly_',
                                 time_path='data_time_', data_cleaner=None, btc_size=16,
                                 time_covariates=True, patch_width=12, block_size=672,
                                 width_factor=2, freq='h', test_horizons=None, scale=True,
                                 is_encoder_model=True, verbose=True):
    """
    Create DataLoader objects of Global Temp data for encoders/decoders training and testing.
    Args:
    - root_path (str): Directory path for dataset files.
    - data_path (str): Filenames with feature data.
    - time_path (str): Filenames with time stamps.
    - data_cleaner (DataCleaning): Data-cleaning pipeline.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - width_factor (float): Number of output patches for encoder.
    Returns:
    - Tuple of DataLoader and scaler objects for train and validate encoders/decoders, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, width_factor, test_horizons)
    size_tv, sizes_te= get_train_val_test_sizes(patch_width, block_size, width_factor, test_horizons, is_encoder_model)

    train_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='train', size=size_tv, features='S', target=0,
        scale=scale, train_scaler=None, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )
    tds_scaler= train_ds.scaler

    val_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='val', size=size_tv, features='S', target=0,
        scale=scale, train_scaler=tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )

    """ DataLoaders """
    train_loader= DataLoader(train_ds, batch_size=btc_size, shuffle=True)
    val_loader  = DataLoader(val_ds,   batch_size=btc_size, shuffle=False)


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader= {}

    for idx, horizon in enumerate(test_horizons):
        size_te= sizes_te[idx]

        test_ds= Dataset_GlobalTemp(
            root_path, data_path, time_path, split='test', size=size_te, features='S', target=0,
            scale=scale, train_scaler=tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
            data_cleaner=data_cleaner, verbose=verbose
        )
        test_loader[horizon]= DataLoader(test_ds,  batch_size=btc_size, shuffle=False)

    return (
        # Train / Val
        train_loader, val_loader, tds_scaler,
        # Test -- Decoders/Encoders
        *[test_loader[int(h)] for h in test_horizons],
    )
