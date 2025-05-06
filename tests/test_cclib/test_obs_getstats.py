import cclib as cc
import pytest
import numpy as np

"""
Tests to perform:
    Input:
        
    Output:
        - test if number of infected is extracted correctly
"""

# test Input

# test Output

def test_ReadInfected():
    filename = './test_files/p0o1/0o1-10_s0.pkl'

    infected = cc.obs_getstats(filename)

    assert infected == 2