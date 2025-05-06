import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
    - Output
        - check if K counts correctly
'''

# check Input


# check Output

def test_Kcounts():
    """
    Network:
    
     0  0    1  1    1  1
     |  | -> |  | -> |  | 
     0--1    0--1    1--1
    """

    H = nx.Graph()
    H.add_edges_from([(0,2),(1,2),(1,3)])
    nx.set_node_attributes(H,{0:{'status':0},
                            1:{'status':1},
                            2:{'status':0},
                            3:{'status':0}})   # connected to no node

    nx.set_node_attributes(H,{0:{'memory':[1.,1.,2.]},
                            1:{'memory':[1.,1.,2.]},
                            2:{'memory':[0.,0.,0.]},
                            3:{'memory':[0.,0.,0.]}})

    nx.set_node_attributes(H,{0:{'threshold':2.},
                            1:{'threshold':2.},
                            2:{'threshold':2.},
                            3:{'threshold':1.}})

    # set model-run parameter
    S = 3
    f = 1                   # update nodes in each step
    C = 1                   # check consensus in each step
    p = 1
    r = 1
    Xp = [1]
    Yp = [1]
    u = 0                   # node changes opinion, never changes acq.

    K_counter = cc.RunModel(H,f,C,p,r,Xp,Yp,u,S,verbose=True,draw=False,DRF=True)

    assert K_counter == [[0,0],[1,1],[0,1],[0,0]]

