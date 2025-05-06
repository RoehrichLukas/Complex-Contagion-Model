from cde import CDE
import pytest
import numpy as np

'''
Tests to perform:
    - Input
    
    - Output
        - check if y0 is considered
        - check if q is considered
        - check if nu is considered
        - check if m is considered
        - check if T is considered
        - check if p is considered
        - check if ds is considered
        - check if Xp is considered
'''

# check Input



# check Output

def test_y0():

    y00 = [0,0,0]
    y01 = [1,1,1]

    q = 0.5
    nu = 0.5
    m = 1
    T = 1
    p = 1
    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y00,0,q,nu,m,T,p,ds,Xp,Yp) != CDE(y01,0,q,nu,m,T,p,ds,Xp,Yp)

def test_q():

    y0 = [0.5,0.3,0.3]

    q0 = 0.1
    q1 = 0.9

    nu = 0.5
    m = 1
    T = 1
    p = 1
    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q0,nu,m,T,p,ds,Xp,Yp) != CDE(y0,0,q1,nu,m,T,p,ds,Xp,Yp)

def test_f():

    y0 = [0.5,0.3,0.3]
    q = 0.5

    f0 = 0.1
    f1 = 0.9

    m = 1
    T = 1
    p = 1
    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q,f0,m,T,p,ds,Xp,Yp) != CDE(y0,0,q,f1,m,T,p,ds,Xp,Yp)

def test_m():

    y0 = [0.5,0.3,0.3]
    q = 0.5
    nu = 0.5

    m0 = 1
    m1 = 2

    T = 1
    p = 1
    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q,nu,m0,T,p,ds,Xp,Yp) != CDE(y0,0,q,nu,m1,T,p,ds,Xp,Yp)

def test_T():

    y0 = [0.5,0.3,0.3]
    q = 0.5
    nu = 0.5
    m = 1

    T0 = 1
    T1 = 2

    p = 1
    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q,nu,m,T0,p,ds,Xp,Yp) != CDE(y0,0,q,nu,m,T1,p,ds,Xp,Yp)

def test_p():

    y0 = [0.5,0.3,0.3]
    q = 0.5
    nu = 0.5
    m = 1
    T = 1

    p0 = 0.1
    p1 = 0.9

    ds = 1
    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q,nu,m,T,p0,ds,Xp,Yp) != CDE(y0,0,q,nu,m,T,p1,ds,Xp,Yp)

def test_ds():

    y0 = [0.5,0.3,0.3]
    q = 0.5
    nu = 0.5
    m = 1
    T = 1
    p = 1

    ds0 = 1
    ds1 = 2

    Xp = np.array([1])
    Yp = np.array([1])

    assert CDE(y0,0,q,nu,m,T,p,ds0,Xp,Yp) != CDE(y0,0,q,nu,m,T,p,ds1,Xp,Yp)

def test_Xp():

    y0 = [0.5,0.3,0.3]
    q = 0.5
    nu = 0.5
    m = 1
    T = 2
    p = 1
    ds = 2

    Xp1 = np.array([1])
    Xp3 = np.array([3])

    Yp = np.array([1])

    assert CDE(y0,0,q,nu,m,T,p,ds,Xp1,Yp) != CDE(y0,0,q,nu,m,T,p,ds,Xp3,Yp)

def test_simplecases():
    """
    Focus on q and f.
    """

    ni1 = 1
    mi0 = 0
    ms0 = 0
    y0 = [ni1,mi0,ms0]

    q0 = 0
    q1 = 1
    f1 = 1
    f05 = 0.5
    m1 = 1
    T2 = 2
    p0 = 0
    ds1 = 1
    Xp1 = np.array([1])
    Yp1 = np.array([1])


    # q = 0, nu = 1
    exp_res = [-1,0,1]
    assert CDE(y0,0,q0,f1,m1,T2,p0,ds1,Xp1,Yp1) == exp_res
    # q = 0, nu = 0.5
    exp_res = [-0.5,0,0.5]
    assert CDE(y0,0,q0,f05,m1,T2,p0,ds1,Xp1,Yp1) == exp_res

    # q = 1, nu = 1
    exp_res = [0,1,0]
    assert CDE(y0,0,q1,f1,m1,T2,p0,ds1,Xp1,Yp1) == exp_res
    # q = 1, nu = 0.5
    exp_res = [0,0.5,0]
    assert CDE(y0,0,q1,f05,m1,T2,p0,ds1,Xp1,Yp1) == exp_res

