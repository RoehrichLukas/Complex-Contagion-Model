import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check if H is networkx.Graph
    - Output
        x what if H has no nodes?
        x what if H has no edges?
        x length of return array must be number of nodes in H
        x neighbour must be drawn uniformly
'''

# check Input

def test_Input():

    Non_Graph = [1,0.4,-3,[1,2,3],True,'test',{0,1,2},{0:1,2:3}]

    for i in range(len(Non_Graph)):
        with pytest.raises((TypeError,ValueError)):
            cc.AssignAcquaintance(Non_Graph[i],checkInput=True)

# check Output

def test_Emptyness():

    # no nodes, no edges
    H = nx.Graph()
    assert len(cc.AssignAcquaintance(H)) == 0

    # nodes, but no edges
    H.add_nodes_from([0,1,2])
    assert np.array_equal(cc.AssignAcquaintance(H),[None,None,None])

def test_LenOut():

    N_arr = np.logspace(1,4,4,dtype=int)
    M = 10
    G = 2
    T = 3
    Xd = [1]
    Yd = [1]

    for N in N_arr:
        p = M/(N*(N-1)/2)
        H = nx.erdos_renyi_graph(N, p)    # random graph
        cc.InitGraph_attr(H,G,T,Xd,Yd)
        assert len(cc.AssignAcquaintance(H)) == N

def test_Uniform(err=1e-2):

    R = 10000
    H = nx.Graph()
    M = 10
    p = 1/M
    _ = np.full(M,M)
    __ = np.arange(M)
    H.add_edges_from(list(zip(_,__)))       # network with M+1 nodes and each nodes is only connected to the M'th node

    samples = np.zeros(R)
    for r in range(R):
        partner = cc.AssignAcquaintance(H)
        samples[r] = partner[-1]            # only look at partners of M'th node
    counts = np.histogram(samples,bins=np.linspace(0,M,M+1))[0]

    dif = abs(p - counts/R)                 # difference to expected probability
    for I in dif:
        assert I <= err                     # see if difference is smaller then error
    

