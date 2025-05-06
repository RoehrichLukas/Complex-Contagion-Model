from cde import psi
import pytest
import numpy as np

'''
Tests to perform:
    - Input
    
    - Output
        x check if output is a positive float smaller 1
        x check if dose probability function is considered (Xp,Yp)
        x check if p is considered
        x check if T is considered
        x check if ds is considered
        x check simple cases
'''

# check Input



# check Output

def test_Output():
        
    R = 50
    for r in range(R):
        m = np.random.rand()*100
        mi = np.random.rand()*m
        ms = np.random.rand()*(m-mi)
        T = np.random.randint(1,20)
        p = np.random.rand()
        ds = np.random.rand()*T
        Xp = np.array([np.random.randint(1,T+1)])
        Yp = np.array([1])

        assert 0 <= psi(m,mi,ms,T,p,ds,Xp,Yp) <= 1

def test_DoseResponseFunction():
    """
    Check if different dose response functions result in different probabilities.
    """

    Xp1 = np.array([1])
    Xp2 = np.array([3])
    Yp = np.array([1])
    
    m = 1
    mi = 0.3
    ms = 0.3
    T = 2
    p = 1
    ds = 2

    assert psi(m,mi,ms,T,p,ds,Xp1,Yp) != psi(m,mi,ms,T,p,ds,Xp2,Yp)

def test_p():
    """
    Check if different p values result in different probabilities.
    """

    p1 = 0.1
    p2 = 0.9

    Xp = np.array([1])
    Yp = np.array([1])
    m = 1
    mi = 0.3
    ms = 0.3
    T = 2
    ds = 2

    assert psi(m,mi,ms,T,p1,ds,Xp,Yp) != psi(m,mi,ms,T,p2,ds,Xp,Yp)

def test_T():
    """
    Check if different T values result in different probabilities.
    """

    T1 = 1
    T2 = 3

    Xp = np.array([1])
    Yp = np.array([1])
    m = 1
    mi = 0.3
    ms = 0.3
    p = 1
    ds = 2

    assert psi(m,mi,ms,T1,p,ds,Xp,Yp) != psi(m,mi,ms,T2,p,ds,Xp,Yp)

def test_ds():
    """
    Check if different ds values result in different probabilities.
    """

    ds1 = 1
    ds2 = 3

    Xp = np.array([1])
    Yp = np.array([1])
    m = 1
    mi = 0.3
    ms = 0.3
    T = 2
    p = 1

    assert psi(m,mi,ms,T,p,ds1,Xp,Yp) != psi(m,mi,ms,T,p,ds2,Xp,Yp)

def test_simpleCases(err=1e-4):
    """
    Check simple cases.
    """

    m1 = 1
    mi0 = 0
    ms0 = 0     # --> PSI = 1
    ms05 = 0.5  # --> PSI = 1/3
    T2 = 2
    p1 = 1
    p05 = 0.5
    ds1 = 1
    ds2 = 2
    Xp1 = np.array([1])
    Yp1 = np.array([1])
    Xp2 = np.array([2])

    exp_res = 1
    assert np.abs(psi(m1,mi0,ms0,T2,p1,ds1,Xp1,Yp1) - exp_res) < err

    exp_res = 2*(p05)**1*(1-p05)**(2-1)*1 + 1*(p05)**2*(1-p05)**(2-2)*1
    assert np.abs(psi(m1,mi0,ms0,T2,p05,ds1,Xp1,Yp1) - exp_res) < err

    exp_res = 2*(p05)**1*(1-p05)**(2-1)*0 + 1*(p05)**2*(1-p05)**(2-2)*1
    assert np.abs(psi(m1,mi0,ms0,T2,p05,ds2,Xp1,Yp1) - exp_res) < err

    exp_res = 2*(p05)**1*(1-p05)**(2-1)*1 + 1*(p05)**2*(1-p05)**(2-2)*1
    assert np.abs(psi(m1,mi0,ms0,T2,p05,ds2,Xp2,Yp1) - exp_res) < err

    exp_res = 2*(1/3)**1*(1-1/3)**(2-1)*1 + 1*(1/3)**2*(1-1/3)**(2-2)*1
    assert np.abs(psi(m1,mi0,ms05,T2,p1,ds1,Xp1,Yp1) - exp_res) < err