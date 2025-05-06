import cclib as cc
import pytest
import numpy as np

"""
Tests to perform:
    Input:
        - check that there are no '.' and only one '_' in the filename
    Output:
        - check if values are extracted correctly for several cases
        ... different values
        ... different number of parameters
"""

# test Input

def test_Input(Non_Str):
    s1 = '1.0-0.1'
    s2 = 'a1_b2_c3'

    for i in range(len(Non_Str)):
        with pytest.raises((TypeError,ValueError)):
            cc.get_param(Non_Str[i],checkInput=True)

    with pytest.raises((TypeError,ValueError)):
        cc.get_param(s1,checkInput=True)                # test '.' in s
        cc.get_param(s2,checkInput=True )               # test multiple '_' in s

# test Output
        
def test_Cases():
    s1 = '1o3-0o0000001-1e3_s1'
    s2 = '1o3-0o0000001-1o134-5o0-0o02-4o000000002'

    assert cc.get_param(s1) == [1.3,1e-7,1000,'s1']
    assert cc.get_param(s2) == [1.3,1e-7,1.134,5.0,0.02,4.000000002]
