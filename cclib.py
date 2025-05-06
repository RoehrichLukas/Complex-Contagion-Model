"""
written 2024 by Lukas Röhrich @ PIK

Main library for my Master Thesis at the Potsdam Institute for Climate Impact Research (PIK) autumn 2023 to spring 2025.
The thesis constructs and investigates a model for a network based opinion dyanmics inspired by the Adaptive Voter Model
[Holme and Newman, 2006] and takes contagaion attributes from [Dodds and Watts, 2004].

In this library, all the core functions can be found. These contain:
    * InitGraph_attr        - initializes a network with neccessary attributes
    * AssignAcquaintance    - chooses contagion partner for each node
    * NodeInteraction       - dose transmission
    * ChangeAcquaintance    - assign new neighbour to a node
    * ChangeStatus          - change status of a node, i.e. infected -> susceptible
    * NodeUpdate            - decides if either ChangeAcquaintance or ChangeStatus is called
    * RunModel              - runs model, only function user needs to call besides InitGraoh_attr
    * CheckConsense         - check if network reached consensus

The model can be run by stating all the parameters, calling InitGraoh_attr and then RunModel. The model auto-terminates if consensus
is reached or the maximum number of steps is reached.
Most of these core functions contain additional flaggs called `verbose` and `checkInput`, set to False by default. `verbose=True` gives
additional information about the subprocesses running in this function and can be used for error handling. `checkInput=True` makes the
function check the type of the input parameters. This is mainly used for the test-environment found in the directory called `tests` (created
for `pytest`). I recommend to keep it turned off for a model run for better performance.

Further down in this library are functions used for analysis.
"""


import numpy as np
import random
import networkx as nx
from collections import Counter     # only needed for observables
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import glob
import plotly.graph_objs as go


def InitGraph_attr(H,T,n,Xd,Yd,verbose=False,checkInput=False):
    """
    Takes an already existing networkx.Graph H and initializes the attributes 'status', 'memory' and 'threshold'.
    The 'status' attribute is set to 1 for the first n nodes and 0 for the rest.
    The 'memory' attribute is set to a list of T entries for each node, all initialized with 0 if the node is not infected.
    If the node is infected, all T entries are set to 1.
    The 'threshold' attribute is set to a random value drawn from the distribution Xd with respective probabilities Yd.

    :param H:   A Graph with nodes.
    :type H:    networkx.Graph
    :param T:   Number of memory entries for each node.
    :type T:    int
    :param n:   Fraction of infected nodes.
    :type n:    float, [0,1]
    :param Xd:  Drawable values for threshold distribution.
    :type Xd:   numpy.ndarray
    :param Yd:  Respective probabilties for values from Xd.
    :type Yd:   numpy.ndarray, sum of all values must be 1
    """

    # check input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        if (not isinstance(T,(int,np.integer))):
            raise TypeError('N,M,T must be of type int.')
        if (not isinstance(n,(int,np.integer,float,np.floating))):
            raise TypeError('n must be of type int or floating.')
        if (not isinstance(Xd,(list,np.ndarray))) or \
            (not isinstance(Yd,(list,np.ndarray))):
            raise TypeError('Xd,Yd must be of type list or numpy.ndarray.')
        if (T<0):
            raise ValueError('N,M,G,T must be a positve integer.')
        if (n<0) or (n>1):
            raise ValueError('n must be a fraction within [0,1].')
        if (np.sum(Yd) != 1):
            raise ValueError(f"Probabilites in Yp must add up to 1. Currently they add up to {np.sum(Yd)} .")
        if (len(Xd) != len(Yd)):
            raise ValueError("Xp and Yp must be of same dimensions.")
    
    N = len(H.nodes)
    N_arr = np.arange(N)

    # - add status as attribute
    initial_infected = 0
    if n != 0: initial_infected = int(-(-N // (1/n)))                          # always rounds up
    status_arr = [1]*initial_infected + [0]*(N-initial_infected)
    attr1 = [{key:value} for key,value in list(zip(['status']*N,status_arr))]
    attr1 = dict(zip(N_arr,attr1))
    nx.set_node_attributes(H,attr1)

    # - add memory-matrix D as attribute with initialized doses
    initial_memory = [[1. for y in range(T)] for x in range(initial_infected)] + [[0. for y in range(T)] for x in range(N-initial_infected)]
    attr2 = [{key:value} for key,value in list(zip(['memory']*N,initial_memory))]
    attr2 = dict(zip(np.arange(N),attr2))
    nx.set_node_attributes(H,attr2)

    # - add threshold as attribute with values drawn from Xd,Yd
    threshold_arr = np.random.choice(Xd,p=Yd,size=N)
    attr3 = [{key:value} for key,value in list(zip(['threshold']*N,threshold_arr))]
    attr3 = dict(zip(np.arange(N),attr3))
    nx.set_node_attributes(H,attr3)


def AssignAcquaintance(H,checkInput=False):
    """
    Assigns a random acquaintance for each node of a given networkx.Graph H.
    A node with degree 0 is assigned to None.

    :param H:   A Graph with nodes.
    :type H:    networkx.Graph
    :return:    Assigned aucqaintances for each node of H. Position in array is index of node.
    :rtype:     np.ndarray
    """

    # Check Input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
    
    N = len(H.nodes)
    partner = np.full(N, None)
    for node in range(N):
        nn = list(H.neighbors(node))
        if len(nn) == 1:
            aa = nn[0]
            partner[node] = aa
        elif len(nn) >= 1:
            aa = random.choice(nn)          # choose random neighbour if more than one available
            partner[node] = aa
    
    return partner

def NodeInteraction(H,node,partner,exposure,Xp,Yp,verbose=False,checkInput=False):
    """    
    Makes the given node of the graph H interact with the given partner.
    Interaction means exposure by the partner if exposure is True and partner is
    infected. If exposed, the node receives a dose drawn form a given distribution.
    Else, the memory is updated with only 0-dose entries.

    :param H:           A Graph with nodes.
    :type H:            networkx.Graph
    :param node:        Given node which interacts
    :type node:         int
    :param partner:     Interaction partner of node.
    :type partner:      int
    :param exposure:    Indicates if exposure happenes.
    :type exposure:     boolean
    :param Xp:          Drawable values for dose distribution.
    :type Xp:           numpy.ndarray
    :param Yp:          Respective probabilties for values from Xp.
    :type Yp:           numpy.ndarray, sum of all values must be 1
    """

    # check input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        if (not isinstance(node,(int,np.integer))):
            raise TypeError('node must be of type int.')
        if (not isinstance(partner,(int,np.integer,type(None)))):
            raise TypeError('partner must be of type int.')
        if (not isinstance(exposure,(bool,np.bool_))):
            raise TypeError('exposure must be of type boolean.')
        if (not isinstance(Xp,(list,np.ndarray))) or \
            (not isinstance(Yp,(list,np.ndarray))):
            raise TypeError('Xp,Yp must be of type list or numpy.ndarray.')
        if (np.sum(Yp) != 1):
            raise ValueError(f"Probabilites in Yp must add up to 1. Currently they add up to {np.sum(Yp)} .")
        if (len(Xp) != len(Yp)):
            raise ValueError("Xp and Yp must be of same dimensions.")

    # - contagion of dose d into D-memory
    # -- prepare new dose
    H.nodes[node]['memory'].pop(0)              # remove oldest entry
    H.nodes[node]['memory'].append(0.)          # add newest entry, default = 0.0
    # -- change entry if exposed
    if exposure and partner != None:            # node is exposed
        if H.nodes[partner]['status'] == 1:     # partner is infected
            H.nodes[node]['memory'][-1] = np.random.choice(Xp,p=Yp,size=1)[0]


def ChangeAcquaintance(H,node,status_arr,verbose=False,checkInput=False):
    """
    Breaks up the edge between the given node in the networkx.Graph H and assigns a new edge to
    another node of H which has the same attribute 'status' than the node itself and the node
    wasn't connected to before.
    If the given node has no acquaintances to begin with, nothing happens.
    If there is no new acquaintance to choose, the node will stay connected to its old acquaintance.

    :param H:               A Graph
    :type H:                networkx.Graph
    :param node:            Index of node with edge to update.
    :type node:             int
    :param status_arr:      Array with the status of all the nodes listed.
    :type status_arr:       numpy.ndarray, [int]
    """

    # - check Input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        if (not isinstance(node,(int,np.integer))):
            raise TypeError('node must be of type int.')
        if (not isinstance(status_arr,(list,np.ndarray))):
            raise TypeError('status_arr must be of type list or numpy.ndarray.')
        try:
            _ = H.nodes[node]['status']
        except: raise ValueError("node must be vertex in H and have the attribute 'status'.")
        if (not (status_arr[node] == H.nodes[node]['status'])):
            raise ValueError("status_arr must have same entry as the 'status' attribute of the node in H.")
    
    node_acq = list(H.neighbors(node))
    if len(node_acq) == 0 : return                              # if node has no acq. -> skip whole function
    acq_old = random.choice(node_acq)                           # choose random neighbour to be removed
    H.remove_edge(node,acq_old)                                 # remove old neighbour
    if verbose : print('\nnode :',node,'acq_old: ',acq_old)
    if verbose : print('status_arr :',status_arr)
    node_status = H.nodes[node]['status']
    possible_partner = np.where(status_arr == node_status)[0]   # get vertices with same status as target
    possible_partner = np.setdiff1d(possible_partner,np.append(list(H.neighbors(node)),node))   # remove target itself and remaining neighbors from this list
    if verbose : print('possible_partner :',possible_partner)
    if len(possible_partner) == 0: acq_new = acq_old            # if there is no other node to connect to -> connect to old node
    else: acq_new = random.choice(possible_partner)             # choose a non-neighbour vertex with same status as target
    if verbose : print('acq_new: ',acq_new)
    H.add_edge(node,acq_new)                                    # add new neighbour with same status



def ChangeStatus(H,node,r,verbose=False,checkInput=False):
    """
    Check if infection threshold of node is reached by infection doses in its memory. If it is reached the node's
    status will be changed to/left at 'infected'. Otherwise it will be 'suceptible'.
    
    :param H:               A Graph with nodes.
    :type H:                networkx.Graph
    :param node:            Index of node with edge to update.
    :type node:             int
    :param r:               recovery probability
    :type r:                float, [0,1]
    :return:                New opinion of the given node.
    :rtype:                 int
    """

    # - check Input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        if (not isinstance(node,(int,np.integer))):
            raise TypeError('node must be of type int.')
        if (not isinstance(r,(int,np.integer,float,np.floating))):
            raise TypeError('r must be of type int or floating.')
        try:
            _ = H.nodes[node]['status']
            __ = H.nodes[node]['threshold']
        except: raise ValueError("node must be vertex in H and have the attributes 'status' and 'threshold'.")
        if (r<0) or (r>1):
            raise ValueError('r must be a probability within [0,1].')

    old_status = H.nodes[node]['status']
    if verbose : print('\nnode :',node,'old_status :',old_status)
    memory = np.array(H.nodes[node]['memory'])
    cumsum = memory.sum()
    threshold = H.nodes[node]['threshold']
    if verbose : print('cumsum memory:',cumsum)
    new_status = int(cumsum >= threshold)
    if (old_status == 1) and (new_status == 0):
        prob = random.random()
        if prob > r:
            new_status = 1                         # agent stays infected with probability (1-r)
    H.nodes[node]['status'] = new_status
    if verbose : print('new_status :',new_status)

    return new_status


def NodeUpdate(H,node,q,r,status_arr,verbose=False,checkInput=False):
    """
    The given node of the network H is updated in one of two ways.
    -> With probability (u), the node breaks up an existing connection to a node and reconnects to
    another node which has the same status as the node itself, it wasn't connected to before.
    -> With probability (1-u), the node changes its attribute 'status' according to the fact if
    the current doses safed in 'memory' cummulatively exceed the nodes threshold

    :param H:           A Graph with nodes.
    :type H:            networkx.Graph
    :param node:        node to be updated
    :type node:         int
    :param q:           Rewiring probability, [Holme and Newman, 2006].
    :type q:            float, [0,1]
    :param r:           recovery probability
    :type r:            float, [0,1]
    :param opinion_arr: List of all the opinions of the nodes.    
    :type opinion_arr:  numpy.ndarray, [int]
    """

    # - check Input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        if (not isinstance(node,(int,np.integer))):
            raise TypeError('node must be of type int.')
        if (not isinstance(q,(int,np.integer,float,np.floating))):
            raise TypeError('u must be of type int or floating.')
        if (not isinstance(status_arr,(list,np.ndarray))):
            raise TypeError('status_arr must be of type list or numpy.ndarray.')
        if (q<0) or (q>1):
            raise ValueError('q must be a probability within [0,1].')
        try:
            nodes_list = list(H.nodes())
            if len(status_arr) != len(nodes_list):
                raise ValueError("Length of status_arr must be the number of nodes in graph H.")
            _ = H.nodes[nodes_list[0]]['status']
            __ = H.nodes[nodes_list[0]]['memory']
            ___ = H.nodes[nodes_list[0]]['threshold']
        except: raise ValueError("H must be networkx.Graph with the attributes 'status','memory' and 'threshold'.")

    # - Iteration through all the nodes
    prob_arr = np.random.rand()
    if len(list(H.neighbors(node))) == 0: 
        #if verbose: counter[0] += 1
        return                                        # skip this node if it has no acquaintances
    # -- determine case with probability u
    if (prob_arr <= q):
    # --- Case 1: Node is stubborn.
        ChangeAcquaintance(H,node,status_arr)
    else:
    # --- Case 2: Node is flexible.
        status_arr[node] = ChangeStatus(H,node,r)

      
def RunModel(H,tau,C,p,r,Xp,Yp,q,S,verbose=False,animation=False,highlight=None,out_runtime=False,DRF=False,count=False):
    '''
    If out_runtime == True, the function returns the number of iterationsteps the model run took to terminate.
    DRF - Dose Response Function, probability p(K) to change attribute after K contacts with infected.
    if count == True, the function returns a time series of the fraction of states for each consensus check interval

    :param H:           Network with node-attributes 'status', 'memory' and 'threshold'.
    :type H:            networkx.Graph
    :param tau:         Probability of Network Update for a single node to occurr [Holme-Newman].
    :type tau:          float, [0,1]
    :param C:           Interval of Consensus checks
    :type C:            int
    :param p:           Exposure probability.
    :type p:            float, [0,1]
    :param r:           recovery probability
    :type r:            float, [0,1]
    :param Xp:          Drawable values for dose distribution.
    :type Xp:           numpy.ndarray
    :param Yp:          Respective probabilties for values from Xp.
    :type Yp:           numpy.ndarray, sum of all values must be 1
    :param q:           Rewiring probability of nodes. Probability of either changing the status or the acquaintance.
    :type q:            float, [0,1]
    :param S:           Number of maximal total time iteration steps of the model.
    :type S:            int
    :param verbose:     Print out additional information.
    :type verbose:      boolean
    :param animation:   Create an animation of the network.
    :type animation:    boolean
    :param highlight:   Nodes to highlight in the animation.
    :type highlight:    list(int)
    :param out_runtime: Return the number of iteration steps the model run took to terminate.
    :type out_runtime:  boolean
    :param DRF:         Dose Response Function, probability p(K) to change attribute after K contacts with infected
    :type DRF:          boolean
    :param count:       Return a time series of the fraction of states for each consensus check interval.
    :type count:        boolean, return list([[S0,I0,R0],[S1,I1,R1],...,[Sn,In,Rn]])

    '''

    # - pull information from network
    N = len(H.nodes)
    if N == 0:
        print('Graph is empty. Can not run model.')
        return
    T = len(H.nodes[0]['memory'])
    status_arr = np.array(list(dict(sorted(dict(H.nodes.data('status')).items())).values()))
    if DRF:
        K_mem = [[False for y in range(T)] for x in range(N)]              # init. with all 0 : [[0,0,..,0],[0,0,..,0],..,[0,0,..,0]]
        K_counter = [[0,0] for i in range(T+1)]                         # K_counter[i] is K=i, K_counter[:][0] is no change, K_counter[:][1] is change
    if count:
        state_counter = np.zeros((S//C+1,3))                              # state_counter[i] is check-number i, state_counter[:][0] is S, state_counter[:][1] is I, state_counter[:][2] is R
    if animation: frames_cache = []

    # iterate over time steps
    for s in range(S+1):
        # - check if consensus state is reached each mutliple of C
        if s%C == 0:
            Consensus = CheckConsense(H)
            if count:
                state_counter[s//C][0] = np.sum(status_arr == 0)/N
                state_counter[s//C][1] = np.sum(status_arr == 1)/N
                state_counter[s//C][2] = np.sum(status_arr == -1)/N
            # -- create frames of network
            if animation:
                frame = plotly_graph_frame(H,status_arr,s,highlight=highlight)
                frames_cache.append(frame)                
            if Consensus :
                break               # break out of s-loop when consensus state is reached
            if verbose:
                print(f"Consensus: {Consensus}\t -> after {s} steps.")

        # - initialize current step of model iteration
        partner = AssignAcquaintance(H)
        prob_arr1 = np.random.rand(N)            # random number for each node to see if interaction occurs
        prob_arr2 = np.random.rand(N)            # random number for each node to see if update occurs
        # - iterate trough all nodes
        for node in range(N):
            # -- dose interaction
            exposure = (prob_arr1[node] <= p)    # see of exposure happens for individual node, boolean
            NodeInteraction(H,node,partner[node],exposure,Xp,Yp)
                
            # -- network update
            if (prob_arr2[node] <= tau): 
                NodeUpdate(H,node,q,r,status_arr)
            #if verbose: print('\ts_new:',H.nodes[node]['status'])
            # -- update K-memory and check if status has changed and increase one of the counters, ignore the initalisation of memory (s>T)
            if DRF:
                # --- roll list and prepare new entry, default 0.0
                K_mem[node].pop(0)
                K_mem[node].append(False)
                # --- if partner is infected, add contact entry
                if partner[node] != None and H.nodes[partner[node]]['status'] == 1:
                    K_mem[node][-1] = True
                if H.nodes[node]['status'] == 1 and s > T+1:
                    K_counter[np.sum(K_mem[node])][1] += 1
                elif H.nodes[node]['status'] == 0 and s > T+1:
                    K_counter[np.sum(K_mem[node])][0] += 1
    
    if Consensus:
        print(f"Consensus: {Consensus}\t -> after {s} steps.")

    if animation:
        print('Creating Animation...')
        frames_to_animation(frames_cache)
    if out_runtime and not(DRF) and not(count): return s
    if DRF and not(out_runtime) and not(count): return K_counter
    if out_runtime and DRF and not(count): return s,K_counter

    if not(out_runtime) and DRF and count: return K_counter,state_counter
    if not(DRF) and out_runtime and count: return s,state_counter
    if not(out_runtime) and not(DRF) and count: return state_counter

    if out_runtime and DRF and count: return s,K_counter,state_counter


def CheckConsense(H,checkInput=False):
    """        
    Checks if the given Network has reached a Consesus State - meaning every node is only connected
    to nodes with the same 'status' attribute as itself.
    The function goes to each node and compares their status with each acquaintance. Consensus is
    assumed until found otherwise. If non-matching status are found, the function breaks out of the
    loops and stops itself.

    :param H:   A Graph with nodes, each with an attribute 'status'
    :type H:    networkx.Graph
    :return:    True if network has Consensus.
    :rtype:     boolean
    """

    # - check Input
    if checkInput:
        if (not isinstance(H,(nx.Graph))):
            raise TypeError('H must be of type networkx.Graph.')
        try:
            nodes_list = list(H.nodes())
            a = H.nodes[nodes_list[0]]['status']
        except: raise ValueError("H must be networkx.Graph with the attributes 'status'.")

    flag = True
    for node in list(H.nodes()):
        node_op = H.nodes[node]['status']
        for acq in list(H.neighbors(node)):
            if node_op != H.nodes[acq]['status']:
                flag = False
                break
        if not(flag): break

    return flag

def graph_to_plotly(G,status_arr,highlight=[]):
    """
    WARNING: Not tested.

    Takes the networkx.Graph properties and transforms them into a data usable for a plotly graph.

    :param G:   A Graph with nodes.
    :type G:    networkx.Graph
    :param fix: Number of nodes that are fixed in position. -> to stabilize the graph in space
    :type fix:  float
    :return:    edge_trace, node_trace
    :rtype:     go.Scatter, go.Scatter
    """
    highlight.sort(reverse=True)
    status_arr = list(status_arr)

    pos_spring = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos_spring[edge[0]]
        x1, y1 = pos_spring[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos_spring[node]
        node_x.append(x)
        node_y.append(y)
    
    # - add node info -> prepare lists
    node_adjacencies = []
    node_text = []
    node_brim = list(np.ones(len(status_arr))*2)
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node: {node}; NN: {str(len(adjacencies[1]))}')

    # place the highlighted nodes on top (aka at the end of the list)
    for node in highlight:       # sort in descending order to not mix up indices while popping
        node_x.append(node_x.pop(node))
        node_y.append(node_y.pop(node))
        node_text.append(node_text.pop(node))
        node_adjacencies.append(node_adjacencies.pop(node))

        status_arr.append(status_arr.pop(node))

        node_brim[node] = 1.0
        node_brim.append(node_brim.pop(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Reds',
            reversescale=False,
            color=[],
            cmin=0,
            cmax=1,
            size=20,
            line=dict(
                colorscale='Blues',
                color=[],
                cmin=0,
                cmax=2,
                width=[]
                ),
            colorbar=dict(
                thickness=15,
                title='Status',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
        
    # - add node info -> assign values
    node_trace.marker.color = status_arr
    node_trace.text = node_text
    node_trace.marker.line.color = node_brim

    # - highlight nodes by changing brim width
    node_brim[-1*len(highlight):] = [8.0]*len(highlight)
    node_trace.marker.line.width = node_brim


    return edge_trace, node_trace


def plotly_graph_frame(G,status_arr,s=None,highlight=[]):
    """
    WARNING: Not tested.

    Takes a networkx.Graph and transforms it into a plotly frame.

    :param G:   A Graph with nodes.
    :type G:    networkx.Graph
    :param s:   Name of the frame.
    :type s:    str
    :return:    frame
    """
    
    edge_trace, node_trace = graph_to_plotly(G,status_arr,highlight=highlight)
    frame = go.Frame(data=[edge_trace, node_trace],
             name=f'{s}',
             layout=go.Layout(
                title='...',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    return frame

def frames_to_animation(frames):
    """
    WARNING: Not tested.

    Takes a list of plotly frames and creates an animation with an interactive slider.

    :param frames:  List of plotly frames.
    :type frames:   list
    """
    # create figure
    fig = go.Figure(data=[frames[0]['data'][0], frames[0]['data'][1]], layout=frames[0]['layout'], frames=frames).update_layout(
            title = f"TEST Title",
            updatemenus=[
                {
                    "buttons": [{"args": [None, {"frame": {"duration": 1000, "redraw": True}}],
                                "label": "Play", "method": "animate",},
                                {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate", "transition": {"duration": 0},},],
                                "label": "Pause", "method": "animate",},],
                    "type": "buttons",
                }
            ],
            # iterate over frames to generate steps... NB frame name...
            sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                    "mode": "immediate",},],
                                "label": f.name, "method": "animate",}
                                for f in frames],"currentvalue":{"prefix": f"Iteration Step:"}}],
            height=800,
            title_x=0.5,

        )
                
    fig.show()
        

#######################################################################################################################
#############################################  Model Observables  #####################################################
#######################################################################################################################

def get_Status(H):
    """
    WARNING: Not tested.

    Fetches a list of all the status of the nodes in graph H.
    """

    return np.array(list(dict(H.nodes.data('status')).values()))


def get_OpinionDistribution(H,verbose=False):
    """
    WARNING: Not tested.

    Figures out which status in a network exist and how strong they are represented.
    """

    N = len(H.nodes())
    if N == 0: return None
    status_arr = get_Status(H)

    # create dictionary and change key names from numbers to strings
    counts = Counter(status_arr)
    if (0 in counts.keys()): counts['0 susceptible'] = counts.pop(0)
    if (1 in counts.keys()): counts['1 infected'] = counts.pop(1)
    if verbose: 
        print("  Unique Value\t|  Total Frequency  |  Relative Frequency")
        print("---------------------------------------------------------")
        key_sorted = sorted(list(counts.keys()))
        for key in key_sorted:
            print(f"{key}\t|\t {counts[key]}\t    |\t    {round(counts[key]/N*100,2)}%")
        print("---------------------------------------------------------")
        print(f"   -> Total:\t|\t {N}\t    |\t    {round(sum(counts.values())/N*100,2)}% ")

    return np.array(list(counts.values()))[np.argsort(list(counts.keys()))]

def get_Degrees(H):
    """
    WARNING: Not tested.

    Fetches the degrees of all nodes and packs them into a list. List index indicates node index.
    """

    tup_list = list(H.degree())                 # list of tuples (index, degree)
    if len(tup_list) == 0: return None

    return np.array([i for _,i in tup_list])    # only return degrees


def get_AverageDegree(H):
    """
    WARNING: Not tested.

    Returns the average degree of graph H.
    """

    degrees = get_Degrees(H)
    if len(degrees) == 0: return None

    return np.sum(degrees)/len(degrees)

def get_Groups(H,consensus=False):
    """
    WARNING: Not tested.

    Fetches all the seperate subgraphs and returns them as nested lists.

    **Example**
    (Node,Status)

    (1,0)     (2,1)
      |       /  \
    (3,0)  (4,1) (5,1)

    >> [[1,3],[2,4,5]],[0,1]

    """

    sub_graphs = nx.connected_components(H)
    res = []
    status = []
    for c in sub_graphs:
        res.append(list(c))
        if consensus: status.append(H.nodes[list(c)[0]]['status'])
    return res,status

def plot_GroupSizeDistribution(H,legend=None,xlim=None):
    """
    WARNING: Not tested.
    """

    # - prepare data
    groups,_ = get_Groups(H)
    if groups == None:
        print('No plot generated. Network did not reach Consensus.')
        return
    sizes = [len(i) for i in groups]
    Y,bins = np.histogram(sizes,bins='auto')
    X = np.convolve(bins,np.ones(2)/2, mode='valid')    # takes the running mean of the bin borders
    n0_ind = np.where(Y != 0)                           # indices of non-0 histogram values

    # - plotting
    plt.figure(figsize=(10,3))
    plt.scatter(X[n0_ind],Y[n0_ind]/np.max(Y),color='black')
    if legend != None:
        plt.scatter(0,0,s=0,label=legend)
        plt.legend()
    plt.xlabel(r'Group Size $s$')
    plt.ylabel(r'Size Frequency $P(s)$')
    plt.xscale('log')
    plt.yscale('log')
    if xlim != None:
        plt.xlim((0,xlim))
    plt.show()

def get_param(s,checkInput=False):
    """
    Takes a string of format '<p1>-<p2>-...-<pn>_<sample_number>' and extracts a list with all
    the parameters as floats and the sample_number as a string. It replaces 'o' in the string
    with a decimal points.

    ** Example **
    get_param('0o1-10-0o1-0o1-1-0_s0')
    >> [0.1, 10.0, 0.1, 0.1, 1.0, 0.0, 's0']

    :param s:   String containing the parameter values and the sample number.
    :type s:    string; see example for form
    """

    if checkInput:
        if (not isinstance(s,(str))):
            raise TypeError('s must be of type str.')
        if ('.' in s):
            raise ValueError("s not allowed to contain '.'")
        if len(s.split('_')) > 2:
            raise ValueError ("s not allowed to contain more than one '_'")


    sample_split = s.replace('o','.').split('_')
    if len(sample_split)>1: sample = sample_split[-1]
    param_split = sample_split[0].split('-')
    param_list = []
    for param in param_split:
        param_list.append(float(param))
    if len(sample_split)>1: param_list.append(sample)

    return param_list

def obs_DFoutput_single(filename,param_names=['n','T','f','p','r','u','version','runtime','consensus'],avg=False):
    """
    Takes a single output file and transform it into a pandas dataframe. The param_names contains all names of the
    parameters. The last 3 entries must be [...,'version','runtime','consensus'].
    If avg == True, all versions with the same parameters are averaged and the 'version' column is omitted in
    the output.

    :param filename:    path to output file, must be of from: '*p_str-*, where p_str is split-string within file for param-config runs
    :type filename:     str
    :param param_names: list of parameter names in right order of param-config
    :type param_names:  list
    :param avg:         True for versions of same param-config to be averaged
    :type avg:          bool
    """
    #p_str = filename.replace('-','/').split('/')[-2]
    p_str = '/..'
    f = open(filename)
    r = f.read().split(p_str)
    if 'Splitting' in r[-1]:                    # truncates the end in case the model run was not terminated
        r[-1] = r[-1].split('Splitting')[0]

    # initialize dictionary for dataframe
    dict = {}
    for param in param_names:
        dict[param] = []

    for paragraph in r[1:]:
        s = paragraph.split('\n')
        param_val = get_param(s[0][1:])
        for i in range(len(param_val)):
            dict[param_names[i]].append(param_val[i])
        last_line = s[-2].replace('\t','').split(' ')
        try:
            dict['runtime'].append(int(last_line[-2]))
            dict['consensus'].append(last_line[1] == 'True')
        except:
            dict['runtime'].append(np.NaN)
            dict['consensus'].append(np.NaN)

    f.close()
    df = pd.DataFrame(dict)
    if avg:
        target_column = 'runtime'
        condition_columns = param_names.copy()
        condition_columns.remove('runtime')
        condition_columns.remove('version')

        # Calculate the mean based on conditions, NaN entries are skipped by default
        mean_values = df.groupby(condition_columns)[target_column].mean()
        #display(mean_values[:20])
        df = mean_values.reset_index()
        #display(df)
    return df

def obs_DFoutput(DIR_PATH,param_names=['n','T','f','p','r','u','version','runtime','consensus'],avg=False):
    """
    Take a dir path and finds all the files with the extension '.out'. It runs through all these files,
    collects the runtimes of all the parameters runs and stores all the data in a dataframe, sorted by
    the parameters given in the param_names list. The last three entries of this list must be
    [...,'version','runtime','consensus'].
    If avg == True, all versions with the same parameters are averaged and the 'version' column is omitted in
    the output.

    :param DIR_PATH:    path to output directory
    :type DIR_PATH:     str
    :param param_names: list of parameter names in right order of param-config
    :type param_names:  list
    :param avg:         True for versions of same param-config to be averaged
    :type avg:          bool
    """
    filename_list = glob.glob(DIR_PATH+'*.out')
    df_tot = pd.DataFrame({})
    for filename in filename_list:
        df = obs_DFoutput_single(filename,param_names=param_names,avg=avg)
        df_tot = pd.concat([df_tot,df],ignore_index=True)
    return df_tot

def obs_getstats(filename):
    """
    Reads the .pkl file at location filename and counts all infected nodes.
    Returns the total number of infected nodes.Also returns the runtime of the model run.

    :param filename:    path to .pkl file
    :type filename:     str
    """
    df = pd.read_pickle(filename)
    infected = 0
    for index,row in df.iterrows():
        N = len(row['Cluster list'])
        if row['Cluster Status'] == 1: infected += N
    try:
        runtime = df.iloc[-1]['runtime']
    except:
        runtime = np.NaN
    if np.isnan(runtime): infected = np.NaN
    return infected,runtime

def obs_DFcluster_single(DIR_PATH,param_names=['n','T','f','p','r','u','version','infected','runtime'],avg=False,fill=False):
    """
    WARNING: Not tested.

    Takes a directory path of a directory containing model runs and converts the .pkl files
    into a dataframe.
    Last entries of param_names must be [...,'version','infected','runtime].

    :param DIR_PATH:    path to output directory, with leading parameter (often p)
    :type DIR_PATH:     str
    :param param_names: list of parameter names in right order of param-config in file name
    :type param_names:  list
    :param avg:         averages over all versions with the same parameter combination
    :type avg:          bool
    :param fill:        autofill missing values with np.NaN
    :type fill:         bool
    """
    param_df = pd.read_pickle(DIR_PATH+'param_setup.pkl')
    filename_list = glob.glob(DIR_PATH+'*.pkl')
    filename_list.sort()

    N = param_df.iloc[0]['N']

    # initialize dictionary for dataframe
    dict = {}
    for param in param_names:
        dict[param] = []

    for filename in filename_list[:-1]:         # last entry is setup_param.pkl
        param_string = filename.split('/')[-1]
        param_val = get_param(param_string[:-4])
        for i in range(len(param_val)):
            dict[param_names[i]].append(param_val[i])
        infected,runtime = obs_getstats(filename)
        dict['infected'].append(infected/N)
        dict['runtime'].append(runtime)

    # create dataframe out of dictionary
    df = pd.DataFrame(dict)

    # auto fill missing parameter combinations with np.NaN
    if fill:
        # get all the possible combinations of parameters
        a = []
        for param in param_names[:-3]:
            a.append(param_df.iloc[0][param])       # creates list with all the scan-param lists [[a1,a2,...],[b1,b2,..],..]
        b = [list(x) for x in np.array(np.meshgrid(*a)).T.reshape(-1,len(a))]       # creates all possible combinations between these lists

        # iterate through all combinations and check if there is already an entry
        for comb in b:
            comb_exists = (df[param_names[:-3]].values == comb).all(axis=1).any()       # True if pramater combination exists
            if not(comb_exists):
                comb.append(np.NaN)
                comb.append(np.NaN)
                comb.append(np.NaN)
                dict = {}
                for i in range(len(param_names)):
                    dict[param_names[i]] = comb[i]
                df.loc[len(df.index)] = dict
    if avg:
        target_columns = ['infected','runtime']     # columns to be averaged
        condition_columns = param_names.copy()      # columns to group by
        condition_columns.remove('runtime')         # remove non-condition columns
        condition_columns.remove('infected')        # remove non-condition columns
        condition_columns.remove('version')         # remove non-condition columns

        # Calculate the mean based on conditions
        mean_values = df.groupby(condition_columns)[target_columns].mean()
        df = mean_values.reset_index()

    return df

def obs_DFcluster(DIR_PATH,param_names=['n','T','f','p','r','u','version','infected'],avg=False,fill=False):
    """
    Take directory path to whole model run, extracts the data in creates a pandas dataframe.
    Is able to average all runs with same model paramater configuration. Also able to fill
    in missing rows if data file is missing but parameter configuration should be available
    due to param_setup.pkl file.

    :param DIR_PATH:    path to output directory
    :type DIR_PATH:     str
    :param param_names: list of parameter names in right order of param-config in file name
    :type param_names:  list
    :param avg:         averages over all versions with the same parameter combination
    :type avg:          bool
    :param fill:        autofill missing values with np.NaN
    :type fill:         bool
    """

    subdir_list = glob.glob(DIR_PATH+'*')
    subdir_list.sort()
    df_tot = pd.DataFrame({})
    for SUBDIR_PATH in subdir_list[1:]:             # ignore first on because it is dir '00output'
        df = obs_DFcluster_single(SUBDIR_PATH+'/',param_names=param_names,avg=avg,fill=fill)
        df_tot = pd.concat([df_tot,df],ignore_index=True)
    return df_tot


def heatmap_fromDF(df,fix_param,x_param,y_param,val_param,vmax=None,show=False,output=False,value=False,save=False,figsize=(11,8)):
    """
    WARNING: Not tested.

    Takes a pandas DataFrame df, filteres the data and plots a heatmap out of it.
    fix_param is a dictionary with all the parameters that should be fixed to a
    specific value and the coresponding value, e.g. fix_param = {'T':10,'p',0.3}.
    x_param and y_param are strings refering to the names of the parameters that
    shall be plotted in the x- and y-axis of the heatmap.
    val_param is a string refering to the name of the paramater that shall be
    plotted as the value in the heatmap.
    If value = True, the values are plotted in the heatmap additionaly.
    """

    # filter by the fixed parameters
    filtered_df = df.copy()
    for key in fix_param.keys():
        filtered_df = filtered_df[filtered_df[key] == fix_param[key]]

    x_list = filtered_df[x_param].unique()
    x_size = len(x_list)
    y_list = filtered_df[y_param].unique()
    y_size = len(y_list)
    x_list.sort()
    y_list.sort()
    val = np.full([y_size,x_size], np.nan)

    for i in range(x_size):
        for j in range(y_size):
            #filtered_row = filtered_df[(filtered_df[y_param] == y_list[j]) & (filtered_df[x_param] == x_list[i]) & (filtered_df['consensus'] == True)]
            if output : 
                filtered_row = filtered_df[(filtered_df[y_param] == y_list[j]) & (filtered_df[x_param] == x_list[i]) & (filtered_df['consensus'] == True)]
            else:
                filtered_row = filtered_df[(filtered_df[y_param] == y_list[j]) & (filtered_df[x_param] == x_list[i])]
            if len(filtered_row) == 1:
                val[j,i] = filtered_row[val_param].values[0]

    if show:
        cmap = mpl.colormaps.get_cmap('Reds')  # viridis is the default colormap for imshow
        cmap.set_bad(color='black')
        fig, ax = plt.subplots(figsize=figsize)
        shw = ax.imshow(val,cmap=cmap,origin='lower',vmin=0,vmax=vmax)
        # Loop over data dimensions and create text annotations.
        if value == True:
            for i in range(len(val)):
                for j in range(len(val[0])):
                    text = ax.text(j, i, round(val[i, j],2),
                                ha="center", va="center", color="black")
        bar = plt.colorbar(shw)
        bar.set_label(val_param,fontsize=14)
        ax.grid(linewidth=0)
        plt.title(str(fix_param))
        plt.xticks(range(x_size),x_list,rotation=45)
        plt.yticks(range(y_size),y_list)
        plt.xlabel(x_param,fontsize=14)
        plt.ylabel(y_param,rotation=0,fontsize=14)
        if save:
            plt.savefig(f'{x_param}_{y_param}_{fix_param}.png')
        plt.show()

    return x_list,y_list,val

def heatmap_animation(df,scan_param,x_param,y_param,val_param,fix_param,vmax=None,value=False):
    """
    WARNING: Not tested.

    Plots a heatmap animation of the given dataframe. It has an autoplay mode and an
    interactive slider.
    """

    # prepare lists and space
    scan_list = df[scan_param].unique()
    x_list = df[x_param].unique()
    y_list = df[y_param].unique()
    x_list.sort()
    y_list.sort()
    scan_size = len(scan_list)
    x_size = len(x_list)
    y_size = len(y_list)
    val_mem  = np.full([scan_size,y_size,x_size], np.nan)

    # extract data from dataframe
    for i in range(scan_size):
        fix_param[scan_param] = scan_list[i]# = {'T':T_fix,'r':r_fix,'f':f_fix,'n':scan_list[i]}
        x_l,y_l,val = heatmap_fromDF(df,fix_param,x_param,y_param,val_param,vmax=vmax,output=False,value=value,show=False)
        # match indices so entries from val are stored in val_mem at correct position
        y_indices = np.isin(y_list, y_l)
        x_indices = np.isin(x_list, x_l)
        indices_a, indices_b = np.where(y_indices)[0], np.where(x_indices)[0]
        for j in range(len(y_l)):
            for k in range(len(x_l)):
                val_mem[i,indices_a[j], indices_b[k]] = val[j, k]

    # Create frames
    frames = []
    for i in range(scan_size):
        if value:
            heatmap_text = np.around(val_mem[i],decimals=3)
        else:
            heatmap_text = np.full(np.shape(val_mem[i]),'')
        heatmap_data = go.Heatmap(z=val_mem[i],zmin=0,zmax=vmax,text=heatmap_text,texttemplate="%{text}",textfont={"color":"black"})
        frame = go.Frame(data=[heatmap_data], name=scan_list[i], layout=go.Layout(title_text=f"{[str(j)+':'+str(fix_param[j]) for j in fix_param if j!=scan_param]} >> {scan_param} : {scan_list[i]}"))
        frames.append(frame)


    if value:
        heatmap_text = np.around(val_mem[0],decimals=3)
    else:
        heatmap_text = np.full(np.shape(val_mem[0]),'')
    fig = go.Figure(
            data=[go.Heatmap(z=val_mem[0],zmin=0,zmax=vmax,colorscale='Reds',colorbar={"title":val_param},
                             text=heatmap_text,texttemplate="%{text}",textfont={"color":"black"})],frames=frames).update_layout(
        title = f"{[str(i)+':'+str(fix_param[i]) for i in fix_param if i!=scan_param]} >> {scan_param} : {scan_list[0]}",
        updatemenus=[
            {
                "buttons": [{"args": [None, {"frame": {"duration": 1000, "redraw": True}}],
                            "label": "Play", "method": "animate",},
                            {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate", "transition": {"duration": 0},},],
                            "label": "Pause", "method": "animate",},],
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",},],
                            "label": f.name, "method": "animate",}
                            for f in frames],"currentvalue":{"prefix": f"{scan_param}="}}],
        height=800,
        yaxis={"title": y_param, "tickvals":list(range(y_size)), "ticktext":y_list},
        xaxis={"title": x_param, "tickvals":list(range(x_size)), "ticktext":x_list, 'side': 'bottom'},
        title_x=0.5,

    )

    fig.show()
    

def heatmap_criticalmass(df,fix_param,x_param,y_param,cm_param='n',target_column='infected_mean',I_threshold=0.6,show=False,value=False,figsize=(11,8),nmax=1):
    """
    WARNING: Not tested.

    Plots a heatmap which shows (x_param,y_param) as (x,y)-axis and the first n-value that leads
    the parameter set to end up with the final infected population above I_threshold.
    fiy_param is a dict which indicates value of interest for all the remaining params.
    ** Example **
    fix_param = {'p':0.1,'T':10,'r':0.5,...}

    If there is no dataset which matched a specific parameter set, the entry in the heatmap will
    be NaN. If there is a dataset which matched the parameter set but the threshold is not reached,
    the value will be -1.
    """

    filtered_df =  df.copy()
    for key in fix_param.keys():
        filtered_df = filtered_df[filtered_df[key] == fix_param[key]]
    filtered_df = filtered_df[filtered_df[cm_param] <= nmax]

    x_list = filtered_df[x_param].unique()
    x_size = len(x_list)
    y_list = filtered_df[y_param].unique()
    y_size = len(y_list)
    x_list.sort()
    y_list.sort()
    val = np.full([y_size,x_size], np.nan)      # create space to store values in

    for i in range(x_size):
        for j in range(y_size):
            filtered_row = filtered_df[(filtered_df[y_param] == y_list[j]) & (filtered_df[x_param] == x_list[i])]
            # check if there is a non-NaN entry in the infected column
            inf_list = filtered_row[target_column].to_numpy()
            if not(np.isnan(inf_list).all()):
                val[j,i] = -1
                filtered_row = filtered_row[filtered_row[target_column] >= I_threshold]
                if len(filtered_row) > 0:
                    val[j,i] = filtered_row.sort_values(by=[cm_param]).iloc[0][cm_param]

    if show:
        cmap = mpl.colormaps.get_cmap('Reds')  # viridis is the default colormap for imshow
        cmap.set_bad(color='black')
        cmap.set_under(color='blue')
        fig, ax = plt.subplots(figsize=figsize)
        shw = ax.imshow(val,origin='lower',vmin=0,vmax=nmax,cmap=cmap)
        # Loop over data dimensions and create text annotations.
        if value == True:
            for i in range(len(val)):
                for j in range(len(val[0])):
                    text = ax.text(j, i, round(val[i, j],2),
                                ha="center", va="center", color="black")
        bar = plt.colorbar(shw)
        bar.set_label('critical mass',fontsize=14)
        ax.grid(linewidth=0)
        plt.title(str(fix_param)+f', threshold infected: {I_threshold}')
        plt.xticks(range(x_size),x_list,rotation=45,fontsize=14)
        plt.yticks(range(y_size),y_list,fontsize=14)
        plt.xlabel(x_param,fontsize=14)
        plt.ylabel(y_param,rotation=0,fontsize=14)
        plt.show()
    
    return x_list,y_list,val


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    pl_colorscale.append([0, 'blue'])

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def criticalmass_animation(df,scan_param,x_param,y_param,fix_param,target_column='infected_mean',I_threshold=0.6,value=False,nmax=1):
    """
    WARNING: Not tested.

    Plots an animation of the critical mass heatmap. Scan_param is the parameter that is varied with a slider.
    """

    # prepare lists and space
    scan_list = df[scan_param].unique()
    x_list = df[x_param].unique()
    y_list = df[y_param].unique()
    x_list.sort()
    y_list.sort()
    scan_size = len(scan_list)
    x_size = len(x_list)
    y_size = len(y_list)
    val_mem  = np.full([scan_size,y_size,x_size], np.nan)

    # extract data from dataframe
    for i in range(scan_size):
        fix_param[scan_param] = scan_list[i]# = {'T':T_fix,'r':r_fix,'f':f_fix,'n':scan_list[i]}
        x_l,y_l,val = heatmap_criticalmass(df,fix_param,x_param,y_param,target_column=target_column,I_threshold=I_threshold,show=False,nmax=nmax)
        # match indices so entries from val are stored in val_mem at correct position
        y_indices = np.isin(y_list, y_l)
        x_indices = np.isin(x_list, x_l)
        indices_a, indices_b = np.where(y_indices)[0], np.where(x_indices)[0]
        for j in range(len(y_l)):
            for k in range(len(x_l)):
                val_mem[i,indices_a[j], indices_b[k]] = val[j, k]

    # Create frames
    frames = []
    for i in range(scan_size):
        if value:
            heatmap_text = np.around(val_mem[i],decimals=3)
        else:
            heatmap_text = np.full(np.shape(val_mem[i]),'')
        heatmap_data = go.Heatmap(z=val_mem[i],zmin=0,zmax=nmax,text=heatmap_text,texttemplate="%{text}",textfont={"color":"black"})
        frame = go.Frame(data=[heatmap_data], name=scan_list[i], layout=go.Layout(title_text=f"{[str(j)+':'+str(fix_param[j]) for j in fix_param if j!=scan_param]} >> {scan_param} : {scan_list[i]}"))
        frames.append(frame)
    

    cmap = mpl.colormaps.get_cmap('Reds')  # viridis is the default colormap for imshow
    #cmap.set_bad(color='black')
    #cmap.set_under(color='blue')

    pl_cmap = matplotlib_to_plotly(cmap, 255)


    if value:
        heatmap_text = np.around(val_mem[0],decimals=3)
    else:
        heatmap_text = np.full(np.shape(val_mem[0]),'')
    fig = go.Figure(
            data=[go.Heatmap(z=val_mem[0],zmin=0,zmax=nmax,colorscale=pl_cmap,colorbar={"title":"critical mass"},
                             text=heatmap_text,texttemplate="%{text}",textfont={"color":"black"})],frames=frames).update_layout(
        title = f"{[str(i)+':'+str(fix_param[i]) for i in fix_param if i!=scan_param]} >> {scan_param} : {scan_list[0]}",
        updatemenus=[
            {
                "buttons": [{"args": [None, {"frame": {"duration": 1000, "redraw": True}}],
                            "label": "Play", "method": "animate",},
                            {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate", "transition": {"duration": 0},},],
                            "label": "Pause", "method": "animate",},],
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",},],
                            "label": f.name, "method": "animate",}
                            for f in frames],"currentvalue":{"prefix": f"{scan_param}="}}],
        height=800,
        yaxis={"title": y_param, "tickvals":list(range(y_size)), "ticktext":y_list},
        xaxis={"title": x_param, "tickvals":list(range(x_size)), "ticktext":x_list, 'side': 'bottom'},
        title_x=0.5,

    )

    fig.show()


#######################################################################################################################
##########################################  Probability Distributions  ################################################
#######################################################################################################################
    
def Generate_Dist(func,*arg,x0=0.01,x1=10,step=0.01):
    '''
    WARNING: Not tested!

    Generate discretized probability distributions for the given function.

    IN:
        func    :   function for porbability distribution
        *arg    :   function arguments
        x0      :   start of value range
        x1      :   end of value range
        step    :   stepsize of increments in value range
    OUT:
        Xp      :   values of PDF
        Yp      :   probabilites of PDF
    '''

    # - generate values of distribution
    num = int((x1-x0)/step + 1)
    Xp = np.linspace(x0,x1,num)
    Yp = func(Xp,*arg)

    # - add missing values to gain unified distribution in range of X   -> WARNING: residual should be small! only necc. to use numpy.random.choice
    if np.sum(Yp) != 1:
        residual = 1 - np.sum(Yp)*step               # calculate whats missing so that prob. add up to 1
        Xp = np.append(Xp,Xp[-1]+step)
        Yp = np.append(Yp*step,residual)
        #print('Check unification:',np.sum(Yp))

    return Xp,Yp

def deltadist(x,d=1):
    """
    WARNING: Not tested.
    """

    # check input
    if (not isinstance(x,(np.ndarray))):
        raise TypeError("x must be of type 'numpy.ndarray'.")
    if (not isinstance(d,(int,np.integer,float,np.floating))):
        raise TypeError("d must be of type int or float.")
    
    y = np.zeros(len(x))
    y[np.argmin(np.abs(x-d))] = 1
    return y


def gauss_norm(x):

    # check input
    if (not isinstance(x,(np.ndarray))):
        raise TypeError("x must be of type 'numpy.ndarray'.")
    
    return 1/(np.sqrt(2*np.pi)) * np.exp(-((x)**2)/(2))

def lognorm(x,mu,sigma):
    """
    WARNING: Not tested.
    """

    # check input
    if (not isinstance(x,(np.ndarray))):
        raise TypeError("x must be of type 'numpy.ndarray'.")
    if (not isinstance(mu,(int,np.integer,float,np.floating))):
        raise TypeError("mu must be of type int or float.")
    if (not isinstance(sigma,(int,np.integer,float,np.floating))):
        raise TypeError("sigma must be of type int or float.")
    
    return gauss_norm((np.log(x)-mu)/sigma)/x/sigma