# ModelName

This repo is the official implementation for the paper: [ModelName: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts](https://arxiv.org/abs/0000.00000).

## Introduction
In this paper, we introduce ModelName, an encoder-only time-series forecasting model with a Mixture-of-Heterogeneous-Experts (MoHE) architecture, where specialized expert modules capture periodicity and seasonal patterns, enabling state-of-the-art performance in long-horizon multivariate forecasting tasks. MoHE combines heterogeneous experts at both the
sequence and patch levels, enhancing specialization while maintaining the scaling benefits observed in standard MoEs to improve computational efficiency and accuracy. We also incorporate a multimodal cross-attention mechanism that integrates external information from exogenous covariates. With this design, ModelName enhances time series representations by capturing interactions between endogenous features and exogenous information.

## Overall Architecture
ModelName: an encoder-only transformer for multivariate time-series forecasting. (a) The input embedding module splits time channels into sequences of channel-independent patch embeddings. (b) The exogenous embedding module projects, fuses, and patches covariates with the input series to produce aligned exogenous patch embeddings. These patches are processed through B stacked Transformer blocks; each block is composed of self-attention, cross-attention, and a (c) Mixture-of-Heterogeneous-Experts (MoHE) layer for conditional, sparsely-activated computation. (d) The patch decoder head projects final embeddings to forecasting horizons.

<p align="center">
<img src=".\figures\model_architecture.png" width="900" height="" alt="" align=center />
</p>

## TODO List
- A caching mechanism
- Pre-training on large-scale heterogeneous time series datasets

## Usage
1. Install Python 3.12+, and then install the dependencies:

```
pip install -r requirements.txt
```

2. We provide Jupyter notebooks with usage examples in the folder "./notebooks/". You can obtain all multivariate datasets from [[Google Drive]](https://drive.google.com/drive/folders/1Nz3qE3-lJmJ758c0wbmiDoqQM6k73Bjr?usp=sharing), and we also provide methods to download them automatically.

3. Train and evaluate a model.

4. You can reproduce the experiment results by downloading our checkpoints from [[Google Drive]](https://drive.google.com/drive/folders/1C6OEebq9k9WLTFY4f69r2cTl6duGadvE?usp=sharing).

## Main Results
We evaluate ModelName on long-term multivariate forecasting benchmarks. Comprehensive forecasting results demonstrate that ModelName effectively incorporates exogenous information to enhance the prediction of endogenous series.

### Full-shot Forecasting

<p align="center">
<img src=".\figures\results.png" width="900" height="" alt="" align=center />
</p>

## Citation
If you find this repo helpful, please cite our paper.

```
@article{xxxx,
  title={ModelName: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts},
  author={Evandro S. Ortigossa, ..., Eran Segal},
  journal={xxxx},
  year={xxxx}
}
```

## Acknowledgement
We appreciate the following GitHub repos for their valuable efforts:

Stationary (https://github.com/thuml/Nonstationary_Transformers)

TimeXer (https://github.com/thuml/TimeXer)

Time-MoE (https://github.com/Time-MoE/Time-MoE)

PatchTST (https://github.com/yuqinie98/PatchTST)

U-ViT (https://github.com/baofff/U-ViT)

## Contact
Please let us know if you have any suggestions or find out a mistake: 
evandro.scudeleti-ortigossa@weizmann.ac.il or eran.segal@weizmann.ac.il or submit an issue.

## License
This project is licensed under the Apache-2.0 License.