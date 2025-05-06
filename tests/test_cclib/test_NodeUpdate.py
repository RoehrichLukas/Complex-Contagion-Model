import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check that H is networkx.Graph with correct attributes
        x check if node is int
        x check that u is probability
        x check if status_arr is numpy.ndarray...
            ... with same length as number of nodes in H
    - Output
        x check if node of H and status_arr is updated correctly in each case
        x check if u is applied statisitcally correct
'''

# check Input

def test_Input(Non_Array,Non_Int,Non_Int_or_Float,Non_Prob,Non_Graph):

    # network without 'memory' attribute
    H1 = nx.Graph()
    H1.add_node(0)
    nx.set_node_attributes(H1,{0:{'status':0}})
    nx.set_node_attributes(H1,{0:{'threshold':1.}})
    # network without 'status' attribute
    H2 = nx.Graph()
    H2.add_node(0)
    nx.set_node_attributes(H2,{0:{'memory':np.array([[0.,0.,0.],[1.,1.,1.]])}})
    nx.set_node_attributes(H2,{0:{'threshold':1.}})
    # network without 'threshold' attribute
    H3 = nx.Graph()
    H3.add_node(0)
    nx.set_node_attributes(H3,{0:{'status':0}})
    nx.set_node_attributes(H3,{0:{'memory':np.array([[0.,0.,0.],[1.,1.,1.]])}})
    # probabilty
    node = 0
    u = 1
    r = 1
    status_arr = np.array([0])

    # - check types
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.NodeUpdate(i,node,u,r,status_arr,checkInput=True)
    for i in Non_Int:
        with pytest.raises((TypeError)):
            cc.NodeUpdate(H1,i,u,r,status_arr,checkInput=True)
    for i in Non_Int_or_Float:
        with pytest.raises((TypeError)):
            cc.NodeUpdate(H1,node,i,r,status_arr,checkInput=True)
    for i in Non_Array:
        with pytest.raises((TypeError)):
            cc.NodeUpdate(H1,node,u,r,i,checkInput=True)
    # - check values
    for i in Non_Prob:
        with pytest.raises(ValueError):
            cc.NodeUpdate(H1,node,i,r,status_arr,checkInput=True)
    with pytest.raises((ValueError)):
        cc.NodeUpdate(H1,node,u,r,status_arr,checkInput=True)
        cc.NodeUpdate(H2,node,u,r,status_arr,checkInput=True)
        cc.NodeUpdate(H3,node,u,r,status_arr,checkInput=True)
    # - status_arr too long
    with pytest.raises((ValueError)):
        cc.NodeUpdate(H1,node,u,r,np.array([0,0]),checkInput=True)


# check Output

def test_cases():
    """
    Cases:
    -> trigger new acquaintance
            - change acq because same opinion
            - do not change acq because no connection can be broken up
            - do not change acq because no node has the same status

    -> trigger new status
            - change status due to threshold
            - do not change status due to threshold

    Initial State of Network (Node,Status):

        (0,0)   (2,0)
          |
        (1,1)
    """

    H = nx.Graph()
    H.add_edges_from([(0,1)])                   # only node 1 is connected to node 0
    H.add_node(2)
    nx.set_node_attributes(H,{0:{'status':0},
                              1:{'status':1},  # node 0 is connected to node 1, which has not the same status
                              2:{'status':0}}) # node 0 is not connected to node 2, which has the same status
    nx.set_node_attributes(H,{0:{'memory':[1.,1.,0.]},
                              1:{'memory':[1.,0.,1.]},
                              2:{'memory':[0.,0.,0.]}})
    nx.set_node_attributes(H,{0:{'threshold':1.},
                              1:{'threshold':1.},
                              2:{'threshold':1.}})
    r = 1
    status_arr = np.array([0,1,0])
    
    #assert np.array_equal(list(H.neighbors(0)),[1])
    # trigger new acquaintance
    u = 1
    # - changes acquaintance because of same status
    cc.NodeUpdate(H,0,u,r,status_arr)
    assert np.array_equal(list(H.neighbors(0)),[2])
    assert H.nodes[0]['status'] == 0
    assert status_arr[0] == 0
    # - does not change acquaintance because it has no partner
    cc.NodeUpdate(H,1,u,r,status_arr)
    assert np.array_equal(list(H.neighbors(1)),[])
    assert H.nodes[1]['status'] == 1
    assert status_arr[1] == 1
    # - does not change acquaintance because there is no other possible node
    cc.NodeUpdate(H,2,u,r,status_arr)
    assert np.array_equal(list(H.neighbors(2)),[0])
    assert H.nodes[2]['status'] == 0
    assert status_arr[2] == 0

    """
    Current State of Network (Node,Status):

        (0,0)--(2,0)
          
        (1,1)
    """

    # trigger new opinion
    u = 0
    # - changes opinion because of threshold
    cc.NodeUpdate(H,0,u,r,status_arr)
    assert np.array_equal(list(H.neighbors(0)),[2])
    assert H.nodes[0]['status'] == 1
    assert status_arr[0] == 1
    # - does not change opinion, because of threshold
    cc.NodeUpdate(H,1,u,r,status_arr)
    assert np.array_equal(list(H.neighbors(1)),[])
    assert H.nodes[1]['status'] == 1
    assert status_arr[1] == 1

    """
    Final State of Network (Node,Status):

        (0,1)--(2,0)
          
        (1,1)
    """

def test_probability_u(err = 1e-2):
    """
    Test idea:
        - make simple network
        - update one specific node (here: 0) with probability u
        - repeat R times
        - compare statistics
    
    Initial State of Network (Node,Status):

        (0,0)   (2,0)
          |
        (1,1)

    """

    u = 0.75
    r = 1
    R = 10000             # number of repetitions to draw samples

    counts = [0,0]
    for r in range(R):
        # initiate network
        H = nx.Graph()
        H.add_edges_from([(0,1)])                   # only node 1 is connected to node 0
        H.add_node(2)
        nx.set_node_attributes(H,{0:{'status':0},
                                1:{'status':1},    # node 0 is connected to node 1, which has not the same status
                                2:{'status':0}})   # node 0 is not connected to node 2, which has the same status
        nx.set_node_attributes(H,{0:{'memory':[1.,1.,0.]},
                                1:{'memory':[1.,0.,1.]},
                                2:{'memory':[0.,0.,0.]}})
        nx.set_node_attributes(H,{0:{'threshold':1.},
                                1:{'threshold':1.},
                                2:{'threshold':1.}})
        status_arr = np.array([0,1,0])

        # apply tested function
        node = 0
        cc.NodeUpdate(H,node,u,r,status_arr)

        if np.array_equal(list(H.neighbors(0)),[2]):    # if new acquaintance
            counts[0] += 1
        if H.nodes[0]['status'] == 1:                  # if new opinion
            counts[1] += 1

    dif_new_acq = counts[0]/R - u
    dif_new_op = counts[1]/R - (1-u)

    assert abs(dif_new_acq) <= err
    assert abs(dif_new_op) <= err

