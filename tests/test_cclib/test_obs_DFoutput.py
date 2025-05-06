import cclib as cc
import pytest
import numpy as np

"""
Tests to perform:
    Input:
        
    Output:
        - check if data is extracted correctly from test files
"""


# test Input

# test Output

def test_ReadData():
    DIR_PATH = './test_files/00output/'
    param_names = ['param1','param2','version','runtime','consensus']

    df = cc.obs_DFoutput(DIR_PATH,param_names=param_names)

    assert len(df) == 5
    assert df.iloc[-1]['runtime'] == 40.0