import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check if H is network.Graph 
        x check if node is integer... 
        x ... which exists in H and has the attribute 'status'
        x check if status_arr is list...
        x ... and its entry for node is the same as the nodes attribute 'status' in H
    - Output
        x check if function does NOT change content of status_arr
        x check that ONLY the given node changes acquaintance
        x check that an acquaintance is choosen which has the same status
        x check that function does not change anything in the network if node does not have any acq.
        x check if node reconnects to old acquaintance if there is no other to choose from
'''

# check Input

def test_Input(Non_Int,Non_Array,Non_Graph):

    H = nx.Graph()
    H.add_node(0)
    H.add_node(1)
    nx.set_node_attributes(H,{1:{'status':1}})

    node = 0
    status_arr = np.array([1,2])      # entry at index 1 is not same as status of node 1


    # - check types
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.ChangeAcquaintance(i,node,status_arr,checkInput=True)
    for i in Non_Int:
        with pytest.raises((TypeError)):
            cc.ChangeAcquaintance(H,i,status_arr,checkInput=True)
    for i in Non_Array:
        with pytest.raises((TypeError)):
            cc.ChangeAcquaintance(H,node,i,checkInput=True)
    # - check values
    with pytest.raises((ValueError)):
        cc.ChangeAcquaintance(H,0,status_arr,checkInput=True)     # node 0 exists, but does not have the attribute 'status'
        cc.ChangeAcquaintance(H,1,status_arr,checkInput=True)     # node 1 has not the same 'status' attribute as given in status_arr
        cc.ChangeAcquaintance(H,2,status_arr,checkInput=True)     # node 2 does not exist


# check Output

def test_OpinionsUnchanged():

    H = nx.Graph()
    H.add_edges_from([(0,1)])                   # only node 1 is connected to node 0
    H.add_node(2)
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':0}}) # node 0 is not connected to node 2, which has the same status
    
    status_arr = np.array(list(dict(H.nodes.data('status')).values()))

    cc.ChangeAcquaintance(H,0,status_arr)
    cc.ChangeAcquaintance(H,1,status_arr)
    cc.ChangeAcquaintance(H,2,status_arr)

    status_arr = np.array(list(dict(H.nodes.data('status')).values()))

    assert np.array_equal(status_arr,np.array([0,1,0]))

def test_OnlyNodeIsAffected():

    H = nx.Graph()
    H.add_edges_from([(0,1),(1,2)])
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':0}}) # node 0 is not connected to node 2, which has the same status

    node = 0
    status_arr = np.array(list(dict(H.nodes.data('status')).values()))
    R = 100

    assert np.array_equal(list(H.neighbors(0)),np.array([1]))
    assert np.array_equal(list(H.neighbors(1)),np.array([0,2]))
    assert np.array_equal(list(H.neighbors(2)),np.array([1]))

    for r in range(R):
        cc.ChangeAcquaintance(H,node,status_arr)    # -> node 0 breaks connection with node 1 and makes new connection to node 2

        assert np.array_equal(list(H.neighbors(0)),np.array([2]))
        assert np.array_equal(list(H.neighbors(1)),np.array([2]))
        assert np.array_equal(list(H.neighbors(2)),np.array([1,0]))

def test_AcqWithSameStatusIsChoosen():

    H = nx.Graph()
    H.add_edges_from([(0,1)])
    H.add_node(2)
    H.add_node(3)
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':0}, # node 0 is not connected to node 2, which has the same status
                              3:{'status':1}}) # node 0 is not connected to node 3, which has not the same status
    
    node = 0
    status_arr = np.array(list(dict(H.nodes.data('status')).values()))
    R = 100

    assert np.array_equal(list(H.neighbors(0)),np.array([1]))       # node 0 should only have node 1 ad neighbour

    for r in range(R):
        cc.ChangeAcquaintance(H,node,status_arr)
        assert np.array_equal(list(H.neighbors(0)),np.array([2]))   # node 0 should never make a connection to any other node than node 2

def test_NodeHasNoAcq():

    H = nx.Graph()
    H.add_edges_from([(0,1)])
    H.add_node(2)
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':0}}) # node 0 is not connected to node 2, which has the same status

    node = 2
    status_arr = np.array(list(dict(H.nodes.data('status')).values()))
    R = 100

    assert np.array_equal(list(H.neighbors(0)),np.array([1]))
    assert np.array_equal(list(H.neighbors(1)),np.array([0]))
    assert np.array_equal(list(H.neighbors(2)),np.array([]))

    for r in range(R):
        cc.ChangeAcquaintance(H,node,status_arr)    # node 2 should never make a new connection because it can not break an old connection

        assert np.array_equal(list(H.neighbors(0)),np.array([1]))
        assert np.array_equal(list(H.neighbors(1)),np.array([0]))
        assert np.array_equal(list(H.neighbors(2)),np.array([]))

def test_NoNewAcqWithSameStatus():

    H = nx.Graph()
    H.add_edges_from([(0,1)])
    H.add_node(2)
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':1}}) # node 0 is not connected to node 2, which has not the same status

    node = 0
    status_arr = np.array(list(dict(H.nodes.data('status')).values()))
    R = 100

    assert np.array_equal(list(H.neighbors(0)),np.array([1]))
    assert np.array_equal(list(H.neighbors(1)),np.array([0]))
    assert np.array_equal(list(H.neighbors(2)),np.array([]))

    for r in range(R):
        cc.ChangeAcquaintance(H,node,status_arr)    # node 0 should maintian its connection to 1 because it can not choose another node with same status

        assert np.array_equal(list(H.neighbors(0)),np.array([1]))
        assert np.array_equal(list(H.neighbors(1)),np.array([0]))
        assert np.array_equal(list(H.neighbors(2)),np.array([]))