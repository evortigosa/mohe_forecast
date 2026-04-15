# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Get DataLoader objects for ETT and other datasets.
"""

import numpy as np
from torch.utils.data import DataLoader
from .DataLoaders import Dataset_ETT, Dataset_Custom, Dataset_GlobalTemp



def loaders_safety_checks(patch_width, block_size, out_width, test_horizons):
    # --- safety checks ---
    if patch_width <= 0 or block_size <= 0:
        raise ValueError("patch_width and block_size must be positive.")
    if patch_width > block_size:
        raise ValueError(f"patch_width ({patch_width}) cannot exceed block_size ({block_size}).")
    if out_width < 0.0:
        raise ValueError("out_width must be a non-negative number.")
    if int(patch_width * out_width) > block_size:
        raise ValueError("patch_width * out_width cannot exceed block_size (would make label_len negative).")

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



"""
ETT DataLoaders
"""


def get_ett_data_loaders(ett_root_path, dataset_name_1, dataset_name_2, from_csv, btc_size,
                         time_covariates, patch_width, block_size, out_width, test_horizons=None):
    """
    Create DataLoader objects for ETTx1, ETTx2, and combined datasets for encoder/decoder training
    and testing.
    Args:
    - ett_root_path (str): Directory path for dataset files.
    - dataset_name_1, dataset_name_2 (str): Filenames (e.g., 'ETTh1.csv', 'ETTh2.csv').
    - from_csv (bool): Whether to read data from CSV files or neuralforecast's LongHorizon.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - out_width (float): Number of output patches for encoder.
    Returns:
    - Tuple of DataLoaders and scaler objects for decoder, encoder, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, out_width, test_horizons)

    """ ----- setup for getting training and val data to feed Decoders ----- """
    INPUT_WIDTH = block_size          # how many past steps you feed into the model
    HISTORY_TAIL= block_size - patch_width
    OUTPUT_WIDTH= patch_width         # predict the next time-patch
    dec_size_tv= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]

    dec_train_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='train', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    dec_val_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='val', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    dec_tds_scaler_1= dec_train_ds_ett_1.scaler


    dec_train_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='train', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    dec_val_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='val', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    dec_tds_scaler_2= dec_train_ds_ett_2.scaler


    """ ----- setup for getting training and val data to feed Encoders ----- """
    # int(patch_width * out_width) == 0, generate train/val for SSL mode (data == target)
    E_HISTORY_TAIL= block_size - int(patch_width * out_width)
    E_OUTPUT_WIDTH= int(patch_width * out_width)  # predict a sequence of time-patches
    enc_size_tv= [INPUT_WIDTH, E_HISTORY_TAIL, E_OUTPUT_WIDTH]

    enc_train_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='train', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    enc_val_ds_ett_1= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='val', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    enc_tds_scaler_1= enc_train_ds_ett_1.scaler


    enc_train_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='train', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    enc_val_ds_ett_2= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='val', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    enc_tds_scaler_2= enc_train_ds_ett_2.scaler


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader_ett_1= {}
    test_loader_ett_2= {}

    for horizon in test_horizons:
        if int(patch_width * out_width) > 0:
            F_INPUT_WIDTH = INPUT_WIDTH
            F_HISTORY_TAIL= 0
            F_OUTPUT_WIDTH= horizon
        else:  # int(patch_width * out_width) == 0, generate test for SSL mode (data == target)
            F_INPUT_WIDTH = horizon
            F_HISTORY_TAIL= horizon
            F_OUTPUT_WIDTH= 0

        size_te= [F_INPUT_WIDTH, F_HISTORY_TAIL, F_OUTPUT_WIDTH]

        test_ds_ett_1= Dataset_ETT(
            root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
        )
        test_loader_ett_1[horizon]= DataLoader(test_ds_ett_1, batch_size=btc_size, shuffle=False)

        test_ds_ett_2= Dataset_ETT(
            root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
        )
        test_loader_ett_2[horizon]= DataLoader(test_ds_ett_2, batch_size=btc_size, shuffle=False)


    """ ETTx1 DataLoaders """
    dec_train_loader_ett_1= DataLoader(dec_train_ds_ett_1, batch_size=btc_size, shuffle=True)
    dec_val_loader_ett_1  = DataLoader(dec_val_ds_ett_1,   batch_size=btc_size, shuffle=False)

    enc_train_loader_ett_1= DataLoader(enc_train_ds_ett_1, batch_size=btc_size, shuffle=True)
    enc_val_loader_ett_1  = DataLoader(enc_val_ds_ett_1,   batch_size=btc_size, shuffle=False)

    """ ETTx2 DataLoaders """
    dec_train_loader_ett_2= DataLoader(dec_train_ds_ett_2, batch_size=btc_size, shuffle=True)
    dec_val_loader_ett_2  = DataLoader(dec_val_ds_ett_2,   batch_size=btc_size, shuffle=False)

    enc_train_loader_ett_2= DataLoader(enc_train_ds_ett_2, batch_size=btc_size, shuffle=True)
    enc_val_loader_ett_2  = DataLoader(enc_val_ds_ett_2,   batch_size=btc_size, shuffle=False)


    return (
        # Decoders
        dec_train_loader_ett_1, dec_val_loader_ett_1, dec_train_loader_ett_2, dec_val_loader_ett_2,
        dec_tds_scaler_1, dec_tds_scaler_2,
        # Encoders
        enc_train_loader_ett_1, enc_val_loader_ett_1, enc_train_loader_ett_2, enc_val_loader_ett_2,
        enc_tds_scaler_1, enc_tds_scaler_2,
        # Test -- Decoders/Encoders
        *[test_loader_ett_1[int(h)] for h in test_horizons],
        *[test_loader_ett_2[int(h)] for h in test_horizons],
    )



"""
Custom DataLoaders
"""


def get_custom_data_loaders(root_path, dataset_name, from_csv, btc_size, time_covariates, patch_width,
                            block_size, out_width, freq='min', test_horizons=None):
    """
    Create DataLoader objects for encoder/decoder training and testing.
    Args:
    - root_path (str): Directory path for dataset files.
    - dataset_name (str): Filenames (e.g., 'ETTh1', 'ETTh2', 'Weather').
    - from_csv (bool): Whether to read data from CSV files or neuralforecast's LongHorizon.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - out_width (float): Number of output patches for encoder.
    Returns:
    - Tuple of 10 DataLoader and scaler objects for decoder, encoder, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, out_width, test_horizons)

    """ ----- setup for getting training and val data to feed Decoders ----- """
    INPUT_WIDTH = block_size          # how many past steps you feed into the model
    HISTORY_TAIL= block_size - patch_width
    OUTPUT_WIDTH= patch_width         # predict the next time-patch
    dec_size_tv= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]

    dec_train_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='train', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    dec_val_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='val', size=dec_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    dec_tds_scaler= dec_train_ds.scaler


    """ ----- setup for getting training and val data to feed Encoders ----- """
    # int(patch_width * out_width) == 0, generate train/val for SSL mode (data == target)
    E_HISTORY_TAIL= block_size - int(patch_width * out_width)
    E_OUTPUT_WIDTH= int(patch_width * out_width)  # predict a sequence of time-patches
    enc_size_tv= [INPUT_WIDTH, E_HISTORY_TAIL, E_OUTPUT_WIDTH]

    enc_train_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='train', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    enc_val_ds= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='val', size=enc_size_tv,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )
    enc_tds_scaler= enc_train_ds.scaler


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader= {}

    for horizon in test_horizons:
        if int(patch_width * out_width) > 0:
            F_INPUT_WIDTH = INPUT_WIDTH
            F_HISTORY_TAIL= 0
            F_OUTPUT_WIDTH= horizon
        else:  # int(patch_width * out_width) == 0, generate test for SSL mode (data == target)
            F_INPUT_WIDTH = horizon
            F_HISTORY_TAIL= horizon
            F_OUTPUT_WIDTH= 0

        size_te= [F_INPUT_WIDTH, F_HISTORY_TAIL, F_OUTPUT_WIDTH]

        test_ds= Dataset_Custom(
            root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te,
            features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
        )
        test_loader[horizon] = DataLoader(test_ds, batch_size=btc_size, shuffle=False)

    """ DataLoaders """
    dec_train_loader= DataLoader(dec_train_ds, batch_size=btc_size, shuffle=True)
    dec_val_loader  = DataLoader(dec_val_ds,   batch_size=btc_size, shuffle=False)

    enc_train_loader= DataLoader(enc_train_ds, batch_size=btc_size, shuffle=True)
    enc_val_loader  = DataLoader(enc_val_ds,   batch_size=btc_size, shuffle=False)

    return (
        # Decoders
        dec_train_loader, dec_val_loader, dec_tds_scaler,
        # Encoders
        enc_train_loader, enc_val_loader, enc_tds_scaler,
        # Test -- Decoders/Encoders
        *[test_loader[int(h)] for h in test_horizons],
    )



"""
Global Temp DataLoaders
"""


def get_global_temp_data_loaders(root_path='./global_temp', data_path='temp_global_hourly_',
                                 time_path='data_time_', data_cleaner=None, btc_size=16,
                                 time_covariates=True, patch_width=12, block_size=672,
                                 out_width=2, freq='h', test_horizons=None, verbose=True):
    """
    Create DataLoader objects of Global Temp for encoder/decoder training and testing.
    Args:
    - root_path (str): Directory path for dataset files.
    - data_path (str): Filenames with feature data.
    - time_path (str): Filenames with time stamps.
    - data_cleaner (DataCleaning): Data-cleaning pipeline.
    - btc_size (int): Batch size.
    - time_covariates (bool): Use time features.
    - patch_width (int): Length of each patch.
    - block_size (int): Input sequence length.
    - out_width (float): Number of output patches for encoder.
    Returns:
    - Tuple of 10 DataLoader and scaler objects for decoder, encoder, and test sets.
    """
    test_horizons= loaders_safety_checks(patch_width, block_size, out_width, test_horizons)

    """ ----- setup for getting training and val data to feed Decoders ----- """
    INPUT_WIDTH = block_size          # how many past steps you feed into the model
    HISTORY_TAIL= block_size - patch_width
    OUTPUT_WIDTH= patch_width         # predict the next time-patch
    dec_size_tv= [INPUT_WIDTH, HISTORY_TAIL, OUTPUT_WIDTH]

    dec_train_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='train', size=dec_size_tv, features='S', target=0,
        scale=True, train_scaler=None, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )
    dec_tds_scaler= dec_train_ds.scaler

    dec_val_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='val', size=dec_size_tv, features='S', target=0,
        scale=True, train_scaler=dec_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )


    """ ----- setup for getting training and val data to feed Encoders ----- """
    # int(patch_width * out_width) == 0, generate train/val for SSL mode (data == target)
    E_HISTORY_TAIL= block_size - int(patch_width * out_width)
    E_OUTPUT_WIDTH= int(patch_width * out_width)  # predict a sequence of time-patches
    enc_size_tv= [INPUT_WIDTH, E_HISTORY_TAIL, E_OUTPUT_WIDTH]

    enc_train_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='train', size=enc_size_tv, features='S', target=0,
        scale=True, train_scaler=None, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )
    enc_tds_scaler= enc_train_ds.scaler

    enc_val_ds= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='val', size=enc_size_tv, features='S', target=0,
        scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )


    """ ----- setup for test data - Encoders/Decoders ----- """
    # forecast horizons: {96, 192, 336, 720}
    test_loader= {}

    for horizon in test_horizons:
        if int(patch_width * out_width) > 0:
            F_INPUT_WIDTH = INPUT_WIDTH
            F_HISTORY_TAIL= 0
            F_OUTPUT_WIDTH= horizon
        else:  # int(patch_width * out_width) == 0, generate test for SSL mode (data == target)
            F_INPUT_WIDTH = horizon
            F_HISTORY_TAIL= horizon
            F_OUTPUT_WIDTH= 0

        size_te= [F_INPUT_WIDTH, F_HISTORY_TAIL, F_OUTPUT_WIDTH]

        test_ds= Dataset_GlobalTemp(
            root_path, data_path, time_path, split='test', size=size_te, features='S', target=0,
            scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
            data_cleaner=data_cleaner, verbose=verbose
        )
        test_loader[horizon]= DataLoader(test_ds,  batch_size=btc_size, shuffle=False)


    """ DataLoaders """
    dec_train_loader= DataLoader(dec_train_ds, batch_size=btc_size, shuffle=True)
    dec_val_loader  = DataLoader(dec_val_ds,   batch_size=btc_size, shuffle=False)

    enc_train_loader= DataLoader(enc_train_ds, batch_size=btc_size, shuffle=True)
    enc_val_loader  = DataLoader(enc_val_ds,   batch_size=btc_size, shuffle=False)

    return (
        # Decoders
        dec_train_loader, dec_val_loader, dec_tds_scaler,
        # Encoders
        enc_train_loader, enc_val_loader, enc_tds_scaler,
        # Test -- Decoders/Encoders
        *[test_loader[int(h)] for h in test_horizons],
    )
