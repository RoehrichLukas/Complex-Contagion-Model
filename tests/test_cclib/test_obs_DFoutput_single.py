import cclib as cc
import pytest
import numpy as np

"""
Tests to perform:
    Input:
        
    Output:
        - check if data is extracted correctly from test file
        - check if average is calculated correctly
"""

# test Input

# test Output

def test_ReadData():
    filename = './test_files/00output/p0o1-test_output1.out'
    param_names = ['param1','param2','version','runtime','consensus']

    df = cc.obs_DFoutput_single(filename,param_names=param_names)

    assert df.iloc[0]['param1'] == 0.1
    assert df.iloc[0]['param2'] == 10
    assert df.iloc[0]['version'] == 's0'
    assert df.iloc[0]['runtime'] == 10
    assert df.iloc[0]['consensus'] == True
    assert df.iloc[2]['param1'] == 0.1
    assert df.iloc[2]['param2'] == 10
    assert df.iloc[2]['version'] == 's2'
    assert df.iloc[2]['runtime'] == 20
    assert df.iloc[2]['consensus'] == False

def test_Average():
    filename = './test_files/00output/p0o1-test_output1.out'
    param_names = ['param1','param2','version','runtime','consensus']

    df = cc.obs_DFoutput_single(filename,param_names=param_names,avg=True)

    assert df.iloc[1]['runtime'] == 15.0