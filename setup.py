"""
The Time-Series Forecasting Transformer (TSFT) with Mixture-of-Heterogeneous-Experts (MoHE) Model
"""

from setuptools import setup, find_packages

setup(
    name='mohe_forecast',
    version='4.0.1',
    description='Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Evandro S. Ortigossa',
    url='https://github.com/evortigosa/mohe_forecast',
    packages=find_packages(),
)
