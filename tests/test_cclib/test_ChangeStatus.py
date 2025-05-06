import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check if H is network.Graph 
        x check if node is integer... 
        x ... which exists in H and has the attribute 'status' and 'threshold'
        x check if r is probabilty
    - Output
        x check if acquaintances are still the same after execution
        x check if status is changed/maintained according to threshold
            -> threshold reached:       - I -> I
                                        - S -> I
            -> threshold not reached:   - S -> S
                                        - I -> S
        x check if function returns correct new status
        x check if only status of given node is changed
        x check if probabilty r is stochastically correct
'''

# check Input

def test_Input(Non_Int,Non_Graph,Non_Prob):

    H = nx.Graph()
    H.add_nodes_from([0,1])
    nx.set_node_attributes(H,{0:{'threshold':1},
                              1:{'status':0}})

    node = 0
    r = 1

    # - check types
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.ChangeStatus(i,node,r,checkInput=True)
    for i in Non_Int:
        with pytest.raises((TypeError)):
            cc.ChangeStatus(H,i,r,checkInput=True)
    for i in Non_Prob:
        with pytest.raises((TypeError,ValueError)):
            cc.ChangeStatus(H,node,i,checkInput=True)
    # - check values
    assert 'threshold' in H.nodes[0].keys()
    assert 'status' in H.nodes[1].keys()
    with pytest.raises((ValueError)):
        cc.ChangeStatus(H,0,r,checkInput=True)       # node 0 exists, but does not have the attribute 'status'
        cc.ChangeStatus(H,1,r,checkInput=True)       # node 1 exists, but does not have the attribute 'threshold'
        cc.ChangeStatus(H,2,r,checkInput=True)       # node 2 does not exist

# check Output

def test_AcqSame():

    H = nx.Graph()
    H.add_edges_from([(1,2),(2,3)])
    H.add_node(0)
    nx.set_node_attributes(H,{0:{'status':0},   # connected to no node
                              1:{'status':1},   # connected to one node
                              2:{'status':0},   # connected to two node
                              3:{'status':1}})  # connected to one node
    
    nx.set_node_attributes(H,{0:{'memory':[1.,1.,1.]},
                              1:{'memory':[1.,1.,1.]},
                              2:{'memory':[0.,0.,0.]},
                              3:{'memory':[0.,0.,0.]}})
    
    nx.set_node_attributes(H,{0:{'threshold':1.},
                              1:{'threshold':1.},
                              2:{'threshold':1.},
                              3:{'threshold':1.}})
    r = 1
    
    for node in range(4):
        _ = cc.ChangeStatus(H,node,r)

        assert np.array_equal(list(H.neighbors(0)),np.array([]))
        assert np.array_equal(list(H.neighbors(1)),np.array([2]))
        assert np.array_equal(list(H.neighbors(2)),np.array([1,3]))
        assert np.array_equal(list(H.neighbors(3)),np.array([2]))

def test_NodeChange():

    H = nx.Graph()
    H.add_edges_from([(1,2),(2,3)])
    H.add_node(0)
    nx.set_node_attributes(H,{0:{'status':0},   # connected to no node
                              1:{'status':1},   # connected to one node
                              2:{'status':0},   # connected to two node
                              3:{'status':1}})  # connected to one node
    
    nx.set_node_attributes(H,{0:{'memory':[0.,1.,0.]},
                              1:{'memory':[0.,0.,1.]},
                              2:{'memory':[0.,0.,0.]},
                              3:{'memory':[0.,0.,0.]}})
    
    nx.set_node_attributes(H,{0:{'threshold':1.},
                              1:{'threshold':1.},
                              2:{'threshold':1.},
                              3:{'threshold':1.}})
    r = 1
    expected_new_status = [1,1,0,0]
    
    for i in range(4):
        node = i
        new_status = cc.ChangeStatus(H,node,r)

        assert new_status == expected_new_status[i]
        assert H.nodes[node]['status'] == expected_new_status[i]

def test_NoOtherNodeChange():

    H = nx.Graph()
    H.add_edges_from([(1,2),(2,3)])
    H.add_node(0)
    
    nx.set_node_attributes(H,{0:{'memory':[1.,1.,1.]},
                              1:{'memory':[1.,1.,1.]},
                              2:{'memory':[0.,0.,0.]},
                              3:{'memory':[0.,0.,0.]}})
    
    nx.set_node_attributes(H,{0:{'threshold':1.},
                              1:{'threshold':1.},
                              2:{'threshold':1.},
                              3:{'threshold':1.}})
    r = 1
    status_arr = np.array([0,1,0,1])
    
    for node in range(4):
        nx.set_node_attributes(H,{0:{'status':0},   # connected to no node
                                1:{'status':1},   # connected to one node
                                2:{'status':0},   # connected to two node
                                3:{'status':1}})  # connected to one node
        _ = cc.ChangeStatus(H,node,r)

        for notnode in range(4):
            if node == notnode: continue
            else:
                assert H.nodes[notnode]['status'] == status_arr[notnode]

def test_RecoveryDistribution(err = 1e-2):

    H = nx.Graph()
    H.add_node(0)
    
    nx.set_node_attributes(H,{0:{'memory':[0.,0.,0.]}})
    nx.set_node_attributes(H,{0:{'threshold':1.}})
    
    r = 0.75                # recovery probabilty
    R = 10000               # number of repetitions to draw samples


    counts = [0,0]          # store [S,I] after function use
    for s in range(R):
        nx.set_node_attributes(H,{0:{'status':1}})
        new_status = cc.ChangeStatus(H,0,r)

        if new_status == 0:
            counts[0] += 1
        elif new_status == 1:
            counts[1] += 1
        else:
            continue
    
    dif_S = counts[0]/R - r
    dif_I = counts[1]/R - (1-r)

    assert abs(dif_S) <= err
    assert abs(dif_I) <= err
