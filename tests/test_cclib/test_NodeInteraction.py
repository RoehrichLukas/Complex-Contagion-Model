import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check if H is networkx.Graph
        x check if node is int
        x check if partner is int
        x check if exposure is bool
        x check if Xp is array
        x check if Yp is array with sum=1
    - Output
        x the node attribute 'status' of the nodes is not changed
        x D-memory is updated correctly
            x right entry is changed if exposure and partner_status=I and partner != None
            x only 0-dose entries if no exposure
        - dose size is drawn according to distribution
'''

# test Input

def test_Input(Non_Int,Non_Bool,Non_Array,Non_Prob,Non_Graph):

    H = nx.Graph()
    node = 0
    partner = 0
    exposure = True
    Xp = [1,2,3,4]
    Yp = [0.25,0.25,0.25,0.25]

    # - check types
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.NodeInteraction(i,node,partner,exposure,Xp,Yp,checkInput=True)
    for i in Non_Int:
        with pytest.raises((TypeError)):
            cc.NodeInteraction(H,i,partner,exposure,Xp,Yp,checkInput=True)
            cc.NodeInteraction(H,node,i,exposure,Xp,Yp,checkInput=True)
    for i in Non_Bool:
        with pytest.raises((TypeError)):
            cc.NodeInteraction(H,node,partner,i,Xp,Yp,checkInput=True)
    for i in Non_Array:
        with pytest.raises(TypeError):
            cc.NodeInteraction(H,node,partner,exposure,i,Yp,checkInput=True)
            cc.NodeInteraction(H,node,partner,exposure,Xp,i,checkInput=True)

    # - check values
    with pytest.raises(ValueError):
        cc.NodeInteraction(H,node,partner,exposure,Xp,[0.25,0.25,0.25,0],checkInput=True)       # probability sum under 1
        cc.NodeInteraction(H,node,partner,exposure,Xp,[0.25,0.25,0.25,0.5],checkInput=True)     # probability sum over 1
        cc.NodeInteraction(H,node,partner,exposure,[1],[0,1],checkInput=True)

# test Output

def test_StatusNotChanged():

    N = 100
    M = int(N*(N-1)/2)      # all possible connections are there
    p = M/(N*(N-1)/2)
    H = nx.erdos_renyi_graph(N, p)    # random graph
    T = 10
    n = 0.5
    Xd = [1]
    Yd = [1]
    cc.InitGraph_attr(H,T,n,Xd,Yd)

    exposure = True
    Xp = [1,2,3,4]
    Yp = [0,1,0,0]
    
    status_init = np.zeros(N)                      # make space to store initial status
    for n in range(N):
        status_init[n] = H.nodes[n]['status']     # store initial status
    for t in range(T):
        node = np.random.randint(N)
        partner = np.random.randint(N)
        cc.NodeInteraction(H,node,partner,exposure,Xp,Yp)
        status_now = np.zeros(N)                   # make space to store new status
        for n in range(N):
            status_now[n] = H.nodes[n]['status']  # store new status
        
        assert np.array_equal(status_init,status_now)

def test_Dmemory():

    H = nx.Graph()
    H.add_nodes_from([0,1,2,3])
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},
                              2:{'status':1},
                              3:{'status':1}})
    nx.set_node_attributes(H,{0:{'memory':[1.,1.,1.]},
                              1:{'memory':[1.,1.,1.]},
                              2:{'memory':[1.,1.,1.]},
                              3:{'memory':[1.,1.,1.]}})
    
    # case of exposure == False
    cc.NodeInteraction(H,0,1,False,[1,2,3],[0,0,1])
    assert np.array_equal(H.nodes[0]['memory'],[1.,1.,0.])

    # case of exposure == True
    # - partner_status == I
    cc.NodeInteraction(H,0,1,True,[1.,2.,3.],[0,0,1])
    assert np.array_equal(H.nodes[0]['memory'],[1.,0.,3.])
    # - partner == None
    cc.NodeInteraction(H,2,None,True,[1.,2.,3.],[0,0,1])
    assert np.array_equal(H.nodes[2]['memory'],[1.,1.,0.])
    # - partner_status == S
    cc.NodeInteraction(H,3,0,True,[1.,2.,3.],[0,0,1])
    assert np.array_equal(H.nodes[3]['memory'],[1.,1.,0.])

def test_DoseDistribution(err=1e-2):

    p = 1
    Xp = [1,2,3]
    Yp = [0.25,0.25,0.5]
    H = nx.Graph()
    H.add_edges_from([(0,1),(0,2),(0,3)])       # all other nodes are only conncected to 0
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},
                              2:{'status':1},
                              3:{'status':1}})
    nx.set_node_attributes(H,{0:{'memory':[0.,0.,0.,0.]},
                              1:{'memory':[0.,0.,0.,0.]},
                              2:{'memory':[0.,0.,0.,0.]},
                              3:{'memory':[0.,0.,0.,0.]}})
    
    R = 10000               # smaple size

    samples = np.zeros(R)
    for r in range(R):
        node = 0
        partner = np.random.choice([1,2,3])
        cc.NodeInteraction(H,node,partner,True,Xp,Yp)
        samples[r] = H.nodes[0]['memory'][-1]
    counts = np.histogram(samples,bins=[1.,2.,3.,4.])[0]

    assert abs(counts[0]/R - Yp[0]) <= err
    assert abs(counts[1]/R - Yp[1]) <= err
    assert abs(counts[2]/R - Yp[2]) <= err
