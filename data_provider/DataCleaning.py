# -*- coding: utf-8 -*-
"""
Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE)
Data-cleaning Pipline
"""

import numpy as np



class DataCleaning:
    """
    Utility Data-cleaning Pipline
    Adapted from https://arxiv.org/abs/2409.16040
    """

    def __init__(self, top_k=1000, zero_threshold=0.2, window_size=128, min_window_seq_len=256,
                 nan_minimum_seq_length=1) -> None:
        # selection / cleaning hyperparams
        self.top_k= int(top_k)
        self.zero_threshold= float(zero_threshold)
        self.window_size= int(window_size)
        self.min_window_seq_len= int(min_window_seq_len)
        self.nan_minimum_seq_length= int(nan_minimum_seq_length)
        # selection_method currently only supports 'total_valid_length'
        self.selection_method= 'total_valid_length'
        self.chosen_idx= None
        self.station_scores= None


    @staticmethod
    def split_seq_by_nan_inf(seq, minimum_seq_length=1):
        # Missing Value Processing
        output = []
        sublist= []
        for num in seq:
            if num is None or np.isnan(num) or np.isinf(num):
                if len(sublist) >= minimum_seq_length:
                    output.append(np.asarray(sublist))
                    sublist= []
            else:
                sublist.append(num)
        if len(sublist) >= minimum_seq_length:
            output.append(np.asarray(sublist))

        return output


    def split_seq_by_window_quality(self, seq, zero_threshold, window_size=128, minimum_seq_length=256):
        # Invalid Observation Processing
        if len(seq) <= window_size:
            flag, info= self.check_sequence(seq, zero_threshold=zero_threshold)
            if flag:
                return [seq.copy()]
            else:
                return []
        i= window_size
        sub_seq = []
        out_list= []
        # sliding by non-overlapping windows of size window_size
        while True:
            if i + window_size > len(seq):
                window_seq= seq[i - window_size: len(seq)]
                i= len(seq)
            else:
                window_seq= seq[i - window_size: i]
            flag, info= self.check_sequence(window_seq, zero_threshold=zero_threshold)
            if flag:
                # keep raw values for later length accumulation
                sub_seq.extend(window_seq.tolist())
            else:
                if len(sub_seq) >= minimum_seq_length:
                    out_list.append(np.asarray(sub_seq))
                sub_seq= []
            if i >= len(seq):
                break
            i += window_size

        if len(sub_seq) >= minimum_seq_length:
            out_list.append(np.asarray(sub_seq))

        return out_list


    @staticmethod
    def check_sequence(seq, zero_threshold):
        seq= np.asarray(seq)
        if seq.ndim != 1:
            raise RuntimeError(f'Dimension of the seq is not equal to 1: {seq.shape}')
        flag= True
        info= {}
        nan_count= int(np.sum(np.isnan(seq)))
        info['nan_count']= nan_count
        if nan_count > 0:
            flag= False
            return flag, info

        inf_count= int(np.sum(np.isinf(seq)))
        info['inf_count']= inf_count
        if inf_count > 0:
            flag= False
            return flag, info

        zero_ratio= float(np.sum(seq == 0) / len(seq))
        info['zero_ratio']= zero_ratio
        if zero_ratio > zero_threshold:
            flag= False
            return flag, info

        if len(seq) >= 2:
            first_diff= seq[1:] - seq[:-1]
            first_diff_zero_ratio= float(np.sum(first_diff == 0) / len(first_diff))
            info['first_diff_zero_ratio']= first_diff_zero_ratio
            if first_diff_zero_ratio > zero_threshold:
                flag= False
                return flag, info
        else:
            info['first_diff_zero_ratio']= 0.0

        if len(seq) >= 3:
            second_diff= seq[2:] - seq[:-2]
            second_diff_zero_ratio= float(np.sum(second_diff == 0) / len(second_diff))
            info['second_diff_zero_ratio']= second_diff_zero_ratio
            if second_diff_zero_ratio > zero_threshold:
                flag= False
                return flag, info
        else:
            info['second_diff_zero_ratio']= 0.0

        return flag, info


    def quality_score_per_channel(self, raw_data):
        # Compute quality score per channel
        T, S, C= raw_data.shape

        station_scores= np.zeros(S, dtype=np.int64)  # total accepted timestamps per station
        for s in range(S):
            series= raw_data[:, s, 0]  # 1D numpy array length T
            # first: split by NaN/Inf into clean segments
            segments= self.split_seq_by_nan_inf(series, minimum_seq_length=self.nan_minimum_seq_length)
            total_accepted_length= 0
            for seg in segments:
                # second: within each clean segment, extract high-quality subsequences
                accepted_subseqs= self.split_seq_by_window_quality(
                    seg,
                    zero_threshold=self.zero_threshold,
                    window_size=self.window_size,
                    minimum_seq_length=self.min_window_seq_len
                )
                # accumulate lengths
                for a in accepted_subseqs:
                    total_accepted_length += len(a)
            station_scores[s]= total_accepted_length

        # choose top_k stations by score (descending). Stable deterministic sort
        order= np.argsort(-station_scores, kind='stable')
        k= min(self.top_k, S)
        chosen_idx= order[:k]
        # produce selected data: keep original time axis (NaNs preserved) for selected stations
        data_selected= raw_data[:, chosen_idx, :].astype(np.float64, copy=False)  # (T, k, C)

        # persist for later use (val/test) if same instance is passed
        self.chosen_idx= np.asarray(chosen_idx, dtype=int)
        self.station_scores= station_scores

        return data_selected, self.chosen_idx
