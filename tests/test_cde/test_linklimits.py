from cde import mim_limits, msm_limits
import pytest
import numpy as np

'''
Tests to perform:
    - Input
    
    - Output
        - test mi_limits
            - trivial case for mi_in and mi_max
            - non-trivial case for mi_min and mi_max
        - test ms_limits
            - trivial case for ms_min and ms_max
            - non-trivial case for ms_min and ms_max
'''

# check Input



# check Output

def test_mi():
    """
    Network Case:
    N = 5
    ni = 3/5
    
    Full Network: m = 10/N = 2
         _________
        |         |
        | /-I-----S
        |/  | \ / |
        I   |  x  |
        |\  | / \ |
        | \-I-----S
        |_________|

    -> for m = 2/N -> mi_min = 0/N (trivial), mi_max = 2/N (trivial)
     e.g.:                     
                                        
                 I-----S        /-I     S
                       |       /  |      
             I      x  |  or  I   |      
                       |          |      
                 I     S          I     S
                               

    -> for m = 8/N -> mi_min = 1/N (non-trivial), mi_max = 3/N (non-trivial)
    e.g..     _________        _________
             |         |      |         |
             |   I-----S      | /-I     S
             |   | \ / |      |/  | \ / |
             I   |  x  |  or  I   |  x  | 
             |   | / \ |      |\  | / \ |
             |   I-----S      | \-I     S
             |_________|      |_________|

    """

    N = 5
    ni = 3/5
    m = 2/N
    mim_min, mim_max = mim_limits(N, ni, m)
    assert round(mim_min,5) == round(0/N/m,5)
    assert round(mim_max,5) == round(2/N/m,5)

    m = 8/N
    mim_min, mim_max = mim_limits(N, ni, m)
    assert round(mim_min,5) == round(1/N/m,5)
    assert round(mim_max,5) == round(3/N/m,5)

def test_ms():
    """
    Network Case:
    N = 5
    ni = 2/5
    
    Full Network: m = 10/N = 2
         _________
        |         |
        | /-I-----S
        |/  | \ / |
        S   |  x  |
        |\  | / \ |
        | \-I-----S
        |_________|

    -> for m = 2/N, mi = 1/N -> ms_min = 0/N (trivial), ms_max = 1/N (trivial)
     e.g.:                     
                                        
                 I-----S          I     S
                 |                |     |
             S   |  x     or  S   |     |
                 |                |     |
                 I     S          I     S
                               

    -> for m = 8/N, mi = 0/N -> ms_min = 2/N (non-trivial), ms_max = 3/N (non-trivial)
    e.g..     _________        _________
             |         |      |         |
             | /-I-----S      | /-I-----S
             |/    \ /        |/    \ / |
             S      x     or  S      x  | 
             |\    / \        |\    / \ |
             | \-I-----S      | \-I     S
             |_________|      |_________|

    """

    N = 5
    ni = 2/5
    m = 2/N
    mi = 1/N
    msm_min, msm_max = msm_limits(N, ni, m, mi)
    assert round(msm_min,5) == round(0/N/m,5)
    assert round(msm_max,5) == round(1/N/m,5)

    m = 8/N
    mi = 0/N
    msm_min, msm_max = msm_limits(N, ni, m, mi)
    assert round(msm_min,5) == round(2/N/m,5)
    assert round(msm_max,5) == round(3/N/m,5)
