import cclib as cc
import pytest
import numpy as np
"""
Tests to perform:
    Input:
        
    Output:
        - check if data extracted correctly
"""

def test_DataExtraction():

    DIR_PATH = './test_files/'
    param_names = ['param1','param2','version','infected']

    df = cc.obs_DFcluster(DIR_PATH,param_names=param_names)

    assert df.iloc[0]['infected'] == 0.2
    assert df.iloc[1]['infected'] == 0.8
    assert df.iloc[2]['infected'] == 0.8
    assert df.iloc[3]['infected'] == 0.2

