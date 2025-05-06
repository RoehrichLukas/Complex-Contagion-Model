import cclib as cc
import pytest
import numpy as np
"""
Tests to perform:
    Input:
        
    Output:
        - test if data is extracted correctly from dir
        - test if averaged correctly
        - test if gaps are filled
"""

# test Input

# test Output

def test_DataExtraction():

    DIR_PATH = './test_files/p0o1/'
    param_names = ['param1','param2','version','infected']

    df = cc.obs_DFcluster_single(DIR_PATH,param_names=param_names)

    assert df.iloc[0]['infected'] == 0.2
    assert df.iloc[1]['infected'] == 0.8

def test_Average():


    DIR_PATH = './test_files/p0o1/'
    param_names = ['param1','param2','version','infected']

    df = cc.obs_DFcluster_single(DIR_PATH,param_names=param_names,avg=True)

    assert df.iloc[0]['infected'] == 0.5

def test_FillGaps():

    DIR_PATH = './test_files/p0o1/'
    param_names = ['param1','param2','version','infected']

    df = cc.obs_DFcluster_single(DIR_PATH,param_names=param_names,fill=True)

    assert str(df.iloc[2]['infected']) == 'nan'
    assert str(df.iloc[3]['infected']) == 'nan'
    assert str(df.iloc[4]['infected']) == 'nan'