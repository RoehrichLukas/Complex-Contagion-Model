"""
Started May 2024
by Lukas Röhrich @ Potsdam Institute for Climate Impact Research

Library for centrality analysis of compley contagion models
"""

import numpy as np
import networkx as nx
import random

def change_seeds(G, threshold, rank):
    """
    WARNING: Not tested.

    Takes a networkx graph with status attributes and updates the seeds (a.k.a. the initially infected nodes)
    according to their centrality rank. The nodes with the highest centrality are selected as seeds
    and their status is set to 1 (infected). Additionally, the status of (threshold -1) random neighbouring
    nodes is also set to 1. See paper [Guilbeaut and Centola, 2021]
    If threshold = 1, only the nodes with the highest centrality are infected and none of their neighbours.

    
    :param G: Network with status attributes
    :type G: networkx.Graph
    :param threshold: Adoption threshold for contagion
    :type threshold: float
    :param rank: List of node index sorted by their centrality rank.
    :type rank: list
    :return: Network with updated seeds
    """

    N = len(G.nodes)
    T = G.nodes[0]['memory'].__len__()

    status_old = np.array(list(dict(sorted(dict(G.nodes.data('status')).items())).values()))
    status_new = np.zeros(N,dtype=int)
    n_count = len(np.argwhere(status_old == 1))

    # iterate through the nodes in the rank list...
    for central_node in rank:
        if n_count <= 0:                  #... until no more nodes can be infected
            break
        # change status of node with highest rank if not already infected
        if status_new[central_node] == 0:
            status_new[central_node] = 1
            n_count -= 1
        # also infect random subset of neighbours of size T-1 if they are not already infected
        neighbors = list(G.neighbors(central_node))
        chosen_neigh = []
        while len(neighbors) > 0 and len(chosen_neigh) < (threshold-1):     # choose random neighbors
            ix = np.random.choice(np.arange(len(neighbors)),size=1)[0]
            chosen_neigh.append(neighbors.pop(ix))
        for ix in chosen_neigh:
            if n_count <= 0:              # stop if there are no more nodes that can be infected
                break
            if status_new[ix] == 0:     # infect only if not already infected
                status_new[ix] = 1
                n_count -= 1

    # set new status attributes in graph G
    attr1 = [{key:value} for key,value in list(zip(['status']*N,status_new))]
    attr1 = dict(zip(np.arange(N),attr1))
    nx.set_node_attributes(G,attr1)

    # - add memory-matrix D as attribute with initialized doses
    initial_memory = [[1. for y in range(T)] if i == 1 else [0. for y in range(T)] for i in status_new]
    attr2 = [{key:value} for key,value in list(zip(['memory']*N,initial_memory))]
    attr2 = dict(zip(np.arange(N),attr2))
    nx.set_node_attributes(G,attr2)


def create_hubs(G,nHubs,fixM=True,MinHubDeg=0.9,verbose=False):
    """
    WARNING: Not tested.
    
    Takes a networkx.Graph network and creates Hubs (nodes with a high degree).
    It only chooses nodes that are susceptible in G.

    :param G:           Network with status attributes
    :type G:            networkx.Graph  
    :param nHubs:       Number of Nodes transformed to Hubs
    :type nHubs:        int
    :param fixM:        flag if number of edges should be preserved
    :type fixM:         boolean
    :param MinHubDeg:   Percentage of nodes a Hubs should be connected to.
    :type MinHubDeg:    float, <1  
    :return:            List of Hub nodes
    :rtype:             list
    """

    # get total number of nodes in G
    N = len(G.nodes())
    # get the list of susceptible nodes of G
    susc_nodes = [node for node in range(N) if G.nodes[node]['status'] == 0]
    # choose nodes that become the hubs
    Hubs = np.random.choice(susc_nodes,nHubs,replace=False)
    # get the degree of the wannabe Hubs
    Hubs_deg = [G.degree(hub) for hub in Hubs]
    if verbose: print(f'Degree Hubs (old):\t{Hubs_deg}')
    # increase degree of wannabe hubs
    for i in range(nHubs):
        while G.degree(Hubs[i]) < int (N*MinHubDeg):
            hubNN = list(G.neighbors(Hubs[i]))
            # choose a random node which is not a neighbor of the wannabe hub
            newNN = random.choice(range(N))
            while newNN in hubNN:
                newNN = random.choice(range(N))
            G.add_edge(Hubs[i],newNN)
            if fixM:
                try:
                    edge = random.choice(list(set(G.edges(newNN))))
                    G.remove_edge(*edge)
                except:
                    continue
    
    Hubs_deg = [G.degree(hub) for hub in Hubs]
    if verbose: print(f'Degree Hubs (new):\t{Hubs_deg}')
    # check number of edges
    if verbose: print(f'Number of Edges: {G.number_of_edges()}')

    return Hubs 
