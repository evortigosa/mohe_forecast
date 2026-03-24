# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Get DataLoader objects for ETT and other datasets.
"""

import numpy as np
from torch.utils.data import DataLoader
from .DataLoaders import Dataset_ETT, Dataset_Custom, Dataset_GlobalTemp



"""
ETT DataLoaders
"""


def get_ett_data_loaders(ett_root_path, dataset_name_1, dataset_name_2, from_csv, btc_size,
                         time_covariates, patch_width, block_size, out_width):
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
    # --- safety checks ---
    if patch_width <= 0 or block_size <= 0:
        raise ValueError("patch_width and block_size must be positive.")
    if patch_width > block_size:
        raise ValueError(f"patch_width ({patch_width}) cannot exceed block_size ({block_size}).")
    if out_width <= 0.0:
        raise ValueError("out_width must be a positive number.")
    if int(patch_width * out_width) > block_size:
        raise ValueError("patch_width * out_width cannot exceed block_size (would make label_len negative).")

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
    F_HISTORY_TAIL= 0
    size_te_96= [INPUT_WIDTH, F_HISTORY_TAIL, 96]

    test_ds_ett_1_96= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te_96,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    test_ds_ett_2_96= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te_96,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )

    size_te_192= [INPUT_WIDTH, F_HISTORY_TAIL, 192]

    test_ds_ett_1_192= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te_192,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    test_ds_ett_2_192= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te_192,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )

    size_te_336= [INPUT_WIDTH, F_HISTORY_TAIL, 336]

    test_ds_ett_1_336= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te_336,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    test_ds_ett_2_336= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te_336,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )

    size_te_720= [INPUT_WIDTH, F_HISTORY_TAIL, 720]

    test_ds_ett_1_720= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_1, from_csv=from_csv, split='test', size=size_te_720,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )
    test_ds_ett_2_720= Dataset_ETT(
        root_path=ett_root_path, data_path=dataset_name_2, from_csv=from_csv, split='test', size=size_te_720,
        features='MS', target='OT', scale=True, timeenc=1, use_time_features=time_covariates
    )

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

    # forecast horizons: {96, 192, 336, 720}
    test_loader_ett_1_96= DataLoader(test_ds_ett_1_96,  batch_size=btc_size, shuffle=False)
    test_loader_ett_2_96= DataLoader(test_ds_ett_2_96, batch_size=btc_size, shuffle=False)

    test_loader_ett_1_192= DataLoader(test_ds_ett_1_192,  batch_size=btc_size, shuffle=False)
    test_loader_ett_2_192= DataLoader(test_ds_ett_2_192, batch_size=btc_size, shuffle=False)

    test_loader_ett_1_336= DataLoader(test_ds_ett_1_336,  batch_size=btc_size, shuffle=False)
    test_loader_ett_2_336= DataLoader(test_ds_ett_2_336, batch_size=btc_size, shuffle=False)

    test_loader_ett_1_720= DataLoader(test_ds_ett_1_720,  batch_size=btc_size, shuffle=False)
    test_loader_ett_2_720= DataLoader(test_ds_ett_2_720, batch_size=btc_size, shuffle=False)


    return (
        # Decoders
        dec_train_loader_ett_1, dec_val_loader_ett_1, dec_train_loader_ett_2, dec_val_loader_ett_2,
        dec_tds_scaler_1, dec_tds_scaler_2,
        # Encoders
        enc_train_loader_ett_1, enc_val_loader_ett_1, enc_train_loader_ett_2, enc_val_loader_ett_2,
        enc_tds_scaler_1, enc_tds_scaler_2,
        # Test -- Decoders/Encoders
        test_loader_ett_1_96, test_loader_ett_1_192, test_loader_ett_1_336, test_loader_ett_1_720,
        test_loader_ett_2_96, test_loader_ett_2_192, test_loader_ett_2_336, test_loader_ett_2_720,
    )



"""
Custom DataLoaders
"""


def get_custom_data_loaders(root_path, dataset_name, from_csv, btc_size, time_covariates, patch_width,
                            block_size, out_width, freq='min'):
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
    # --- safety checks ---
    if patch_width <= 0 or block_size <= 0:
        raise ValueError("patch_width and block_size must be positive.")
    if patch_width > block_size:
        raise ValueError(f"patch_width ({patch_width}) cannot exceed block_size ({block_size}).")
    if out_width <= 0.0:
        raise ValueError("out_width must be a positive number.")
    if int(patch_width * out_width) > block_size:
        raise ValueError("patch_width * out_width cannot exceed block_size (would make label_len negative).")

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
    F_HISTORY_TAIL= 0
    size_te_96= [INPUT_WIDTH, F_HISTORY_TAIL, 96]

    test_ds_96= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te_96,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )

    size_te_192= [INPUT_WIDTH, F_HISTORY_TAIL, 192]

    test_ds_192= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te_192,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )

    size_te_336= [INPUT_WIDTH, F_HISTORY_TAIL, 336]

    test_ds_336= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te_336,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )

    size_te_720= [INPUT_WIDTH, F_HISTORY_TAIL, 720]

    test_ds_720= Dataset_Custom(
        root_path=root_path, data_path=dataset_name, from_csv=from_csv, split='test', size=size_te_720,
        features='MS', target='OT', scale=True, timeenc=1, freq=freq, use_time_features=time_covariates
    )

    """ DataLoaders """
    dec_train_loader= DataLoader(dec_train_ds, batch_size=btc_size, shuffle=True)
    dec_val_loader  = DataLoader(dec_val_ds,   batch_size=btc_size, shuffle=False)

    enc_train_loader= DataLoader(enc_train_ds, batch_size=btc_size, shuffle=True)
    enc_val_loader  = DataLoader(enc_val_ds,   batch_size=btc_size, shuffle=False)

    # forecast horizons: {96, 192, 336, 720}
    test_loader_96 = DataLoader(test_ds_96,  batch_size=btc_size, shuffle=False)
    test_loader_192= DataLoader(test_ds_192, batch_size=btc_size, shuffle=False)
    test_loader_336= DataLoader(test_ds_336, batch_size=btc_size, shuffle=False)
    test_loader_720= DataLoader(test_ds_720, batch_size=btc_size, shuffle=False)

    return (
        # Decoders
        dec_train_loader, dec_val_loader, dec_tds_scaler,
        # Encoders
        enc_train_loader, enc_val_loader, enc_tds_scaler,
        # Test -- Decoders/Encoders
        test_loader_96, test_loader_192, test_loader_336, test_loader_720,
    )



"""
Global Temp DataLoaders
"""


def get_global_temp_data_loaders(root_path='./global_temp', data_path='temp_global_hourly_',
                                 time_path='data_time_', data_cleaner=None, btc_size=16,
                                 time_covariates=True, patch_width=12, block_size=672,
                                 out_width=2, freq='h', verbose=True):
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
    # --- safety checks ---
    if patch_width <= 0 or block_size <= 0:
        raise ValueError("patch_width and block_size must be positive.")
    if patch_width > block_size:
        raise ValueError(f"patch_width ({patch_width}) cannot exceed block_size ({block_size}).")
    if out_width <= 0.0:
        raise ValueError("out_width must be a positive number.")
    if int(patch_width * out_width) > block_size:
        raise ValueError("patch_width * out_width cannot exceed block_size (would make label_len negative).")

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
    F_HISTORY_TAIL= 0
    size_te_96= [INPUT_WIDTH, F_HISTORY_TAIL, 96]

    test_ds_96= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='test', size=size_te_96, features='S', target=0,
        scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )

    size_te_192= [INPUT_WIDTH, F_HISTORY_TAIL, 192]

    test_ds_192= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='test', size=size_te_192, features='S', target=0,
        scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )

    size_te_336= [INPUT_WIDTH, F_HISTORY_TAIL, 336]

    test_ds_336= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='test', size=size_te_336, features='S', target=0,
        scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )

    size_te_720= [INPUT_WIDTH, F_HISTORY_TAIL, 720]

    test_ds_720= Dataset_GlobalTemp(
        root_path, data_path, time_path, split='test', size=size_te_720, features='S', target=0,
        scale=True, train_scaler=enc_tds_scaler, timeenc=1, freq=freq, use_time_features=time_covariates,
        data_cleaner=data_cleaner, verbose=verbose
    )

    """ DataLoaders """
    dec_train_loader= DataLoader(dec_train_ds, batch_size=btc_size, shuffle=True)
    dec_val_loader  = DataLoader(dec_val_ds,   batch_size=btc_size, shuffle=False)

    enc_train_loader= DataLoader(enc_train_ds, batch_size=btc_size, shuffle=True)
    enc_val_loader  = DataLoader(enc_val_ds,   batch_size=btc_size, shuffle=False)

    # forecast horizons: {96, 192, 336, 720}
    test_loader_96 = DataLoader(test_ds_96,  batch_size=btc_size, shuffle=False)
    test_loader_192= DataLoader(test_ds_192, batch_size=btc_size, shuffle=False)
    test_loader_336= DataLoader(test_ds_336, batch_size=btc_size, shuffle=False)
    test_loader_720= DataLoader(test_ds_720, batch_size=btc_size, shuffle=False)

    return (
        # Decoders
        dec_train_loader, dec_val_loader, dec_tds_scaler,
        # Encoders
        enc_train_loader, enc_val_loader, enc_tds_scaler,
        # Test -- Decoders/Encoders
        test_loader_96, test_loader_192, test_loader_336, test_loader_720,
    )
