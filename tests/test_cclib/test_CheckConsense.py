import cclib as cc
import pytest
import numpy as np
import networkx as nx

'''
Tests to perform:
    - Input
        x check that H is networkx.Graph
        x ...with attribute status
    - Output
        x check if function gives true if network in consense
        - check if function gives false for different cases of non consensus
'''

# check Input

def test_Input(Non_Graph):

    # network without 'status' attribute
    H = nx.Graph()
    H.add_node(0)
    nx.set_node_attributes(H,{0:{'memory':np.array([[0.,0.,0.],[1.,1.,1.]])}})

    # - check types
    for i in Non_Graph:
        with pytest.raises((TypeError)):
            cc.CheckConsense(i,checkInput=True)
    # - check values
    with pytest.raises((ValueError)):
        cc.CheckConsense(H,checkInput=True)


# check Output

def test_ConsensusTrue():
    """
    Network (Node,Status):

       (1,0)     (3,1)
        /  \
    (0,0)  (2,0) (4,0)

    """

    H = nx.Graph()
    H.add_nodes_from([0,1,2,3,4])
    H.add_edges_from([(0,1),(1,2)])
    nx.set_node_attributes(H,{
                           0:{'status':0},     # connected to 1
                           1:{'status':0},     # connected to 0,2
                           2:{'status':0},     # connected to 1
                           3:{'status':1},     # connected to None
                           4:{'status':0},     # connected to None
                           })
    
    assert cc.CheckConsense(H) == True


def test_ConsensusFalse():
    """
    Network (Node,Status):

        (1,1)     (3,1)
        /  \
    (0,0)  (2,0)

    """

    # basic network
    H = nx.Graph()
    H.add_nodes_from([0,1,2,3])
    H.add_edges_from([(0,1),(1,2)])
    nx.set_node_attributes(H,{
                           0:{'status':0},     # connected to 1
                           1:{'status':1},     # connected to 0,2
                           2:{'status':0},     # connected to 1
                           3:{'status':1},     # connected to None
                           })
    
    assert cc.CheckConsense(H) == False

    # connect 1 with one node that has same opinion
    H.add_edges_from([(1,3)])
    assert cc.CheckConsense(H) == False

    # set majority of acq. of 1 opinions to same as 1
    H.nodes[0]['status'] = 2
    assert cc.CheckConsense(H) == False

