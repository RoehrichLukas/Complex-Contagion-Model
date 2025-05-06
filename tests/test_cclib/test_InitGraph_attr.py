import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x H is networkx.Graph
        x T only is positve integer
        x n is fraction
        x Xd is array
        x Yd is array with sum=1 and has same length as Xd
    - Output
        x number of vertices N stays the same
        x number of edges M stays the same
        - edges are distributed uniformly
        x each node has a status attribute, a memory attribute and a threshold attribute
        x initial fraction of infected is n
        x check if D-memory is initialised correctly, init. dose only for infected
        x check if thresholds are drawn according to distribution
'''

# test Input

def test_Input(Non_Int,Non_Pos,Non_Prob,Non_Array,Non_Graph):
    M = 10      # number of edges
    N = 10      # number of vertices
    p = M/(N*(N-1)/2)
    H = nx.erdos_renyi_graph(N, p)    # random graph

    T = 3
    n = 1
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]


    # test for networkx.Graph
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.InitGraph_attr(i,T,n,Xd,Yd,checkInput=True)     # test H argument
    # test for integers value
    for i in Non_Int:
        with pytest.raises((TypeError,ValueError)):
            cc.InitGraph_attr(H,i,n,Xd,Yd,checkInput=True)     # test T argument
    for i in Non_Pos:
        with pytest.raises((TypeError,ValueError)):
            cc.InitGraph_attr(H,i,n,Xd,Yd,checkInput=True)     # test T argument
    # test if n is fraction
    for i in Non_Prob:
        with pytest.raises((TypeError,ValueError)):
            cc.InitGraph_attr(H,T,i,Xd,Yd,checkInput=True)
    # test if Xd,Yd are array
    for i in Non_Array:
        with pytest.raises(TypeError):
            cc.InitGraph_attr(H,T,n,i,Yd,checkInput=True)      # test Xd argument
            cc.InitGraph_attr(H,T,n,Xd,i,checkInput=True)      # test Yd argument

    # - check properties of Yd
    with pytest.raises(ValueError):
        cc.InitGraph_attr(H,T,n,Xd,[0.25,0.25,0.25,0],checkInput=True)         # probability sum under 1
        cc.InitGraph_attr(H,T,n,Xd,[0.25,0.25,0.25,0.5],checkInput=True)       # probability sum over 1
        cc.InitGraph_attr(H,T,n,[1],[0,1],checkInput=True)                     # Xd,Yd not same length

# test Output

def test_Vertices():
    '''
    Test if number of vertices stays the same.
    '''

    N_arr = np.logspace(1,4,4,dtype=int)
    M = 10
    T = 3
    n = 1
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]

    for N in N_arr:
        p = M/(N*(N-1)/2)
        H = nx.erdos_renyi_graph(N, p)    # random graph
        cc.InitGraph_attr(H,T,n,Xd,Yd)
        assert len(H.nodes()) == N

def test_EdgesNum():
    '''
    Test if number of edges stays the same.
    '''

    N = 500
    M_arr = np.logspace(1,4,4,dtype=int)
    T = 3
    n = 1
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]

    for M in M_arr:
        p = M/(N*(N-1)/2)
        H = nx.erdos_renyi_graph(N, p)    # random graph
        M_old = len(H.edges())
        cc.InitGraph_attr(H,T,n,Xd,Yd)
        assert len(H.edges()) == M_old

def test_EdgesUniform():

    ...


def test_NodeAttributes():
    '''
    Test if each node has an attribute called 'status', 'memory' and 'threshold'.
    '''

    N = 10
    M = 10
    p = M/(N*(N-1)/2)
    H = nx.erdos_renyi_graph(N, p)    # random graph
    T = 3
    n = 1
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]

    cc.InitGraph_attr(H,T,n,Xd,Yd)

    # check for each note if it has an attribute called 'status', 'memory' and 'threshold'
    for node in range(N):
        assert ('status' in H.nodes[node].keys())
        assert ('memory' in H.nodes[node].keys())
        assert ('threshold' in H.nodes[node].keys())


def test_InitialInfected():
    '''
    Test if the initial fraction of infected is n.
    '''

    N = 10
    M = 10
    p = M/(N*(N-1)/2)
    H = nx.erdos_renyi_graph(N, p)    # random graph
    T = 3
    n_arr = [0.00001,0.1,0.5,0.9,0.99999]
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]

    expected_infected = [1,1,5,9,10]

    for i in range(len(n_arr)):
        cc.InitGraph_attr(H,T,n_arr[i],Xd,Yd)
        infected_counter = 0
        for node in range(N):
            if H.nodes[node]['status'] == 1:
                infected_counter += 1
        assert infected_counter == expected_infected[i]


def test_Dmemory():
    '''
    Test if D-memory is initialised correctly.
    '''

    N = 10
    M = 10
    p = M/(N*(N-1)/2)
    H = nx.erdos_renyi_graph(N, p)    # random graph
    T = 3
    n = 0.5
    Xd = [1,2,3,4]
    Yd = [0.25,0.25,0.25,0.25]

    cc.InitGraph_attr(H,T,n,Xd,Yd)
    for node in range(N):
        if H.nodes[node]['status'] == 1:
            assert H.nodes[node]['memory'] == [1.,1.,1.]
        else:
            assert H.nodes[node]['memory'] == [0.,0.,0.]


def test_ThresholdDistribution(err=1e-2):

    N = 1
    p = 0
    H = nx.erdos_renyi_graph(N, p)    # random graph
    T = 3
    n = 1
    Xd = [1,2,3]
    Yd = [0.25,0.25,0.5]

    R = 10000               # sample size

    samples = np.zeros(R)
    for r in range(R):
        cc.InitGraph_attr(H,T,n,Xd,Yd)
        samples[r] = H.nodes[0]['threshold']
    counts = np.histogram(samples,bins=[1.,2.,3.,4.])[0]

    assert abs(counts[0]/R - Yd[0]) <= err
    assert abs(counts[1]/R - Yd[1]) <= err
    assert abs(counts[2]/R - Yd[2]) <= err

