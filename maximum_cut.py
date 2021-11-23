# Copyright 2019 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ Import necessary packages ----
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# from graph_creation import compexact_bipartipe_graph, stochastic_block_model
from greedy_algAlix import greedy_algorithm


def compexact_bipartipe_graph(N):
    ## This function takes as arguments: 
    ##  N : number of nodes to put in the graph
    ## This function create a Nx directed graph G which has a complete bipartipe structure. The direction of the edge is from 
    ## A to B, where nodes 1 to N/2 belongs to A and nodes N/2+1 to N belong to B. 
    ## This function returns: 
    ##  G: the graph created
    ## and draws the graph created with label on the nodes
        
    # Create empty graph
    G = nx.DiGraph()
    # Add nodes to the graph
    for i in range(1,N):
        G.add_nodes_from([i,i+1])

    Nhalf = int(N/2)
    # Loop to allocate vertices to nodes
    for i in range(1,Nhalf+1): # Nodes from 0,1,..N/2 belongs to A ...
        for j in range(Nhalf+1,N+1):  #...and nodes from N/2+1,...,N belongs to B. 
            G.add_edges_from([(i,j)]) #add a directed edge from i to j
 
    # Draw the graph created
    setA = list(range(1,Nhalf+1))
    pos=nx.spring_layout(G) # Draw the directed edges
    nx.draw_networkx(G,pos=nx.bipartite_layout(G, setA)) #Draw the directed edges and nodes
    labels = nx.get_edge_attributes(G,'weight') #create the labels (name of the nodes)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels) #add the label to the plot
    filename = "completeproblem.png"
    plt.savefig(filename, bbox_inches='tight')
    return(G)

def stochastic_block_model(N,a=0.1,b=0.5,c=0.1,d=0.1):
    ## This function takes as arguments: 
    ##  N : number of nodes to put in the graph
    ##  a : Probability to have a directed edge between two nodes in A
    ##  b : Probability to have a directed edge between a node in A and a node in B 
    ##  c : Probability to have a directed edge between a node in B and a node in A 
    ##  d : Probability to have a directed edge between two nodes in B
    ## This function create a Nx directed graph G which has a non exact bipartipe structure. Nodes 1 to N/2 (N pair)
    ## belongs to A and nodes N/2+1 to N belongs to B.  
    ## This function returns: 
    ##  G: the graph created
    ## and draws the graph created with label on the nodes
    
 
    # Create empty graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(1,N):
        G.add_nodes_from([i,i+1])
    
    Nhalf = int(N/2)
        
    # Loop to allocate vertices to nodes
    for i in range(1,Nhalf+1): # Nodes from 0,1,..N/2 belongs to A ...
        for j in range(Nhalf+1,N+1):  #...and nodes from N/2+1,...,N belongs to B. 
            pAB = np.random.uniform(0,1) #get a random draw bw 0 and 1
            if pAB>(1-b): #if below the probality that a node goes from A to B
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
        for j in range(1,Nhalf+1): #... and nodes from 0,1,...,N/2+1 belongs to A
            pAA = np.random.uniform(0,1) 
            if (j!=i) and pAA>(1-a): # additional conditions is i different from j (do not want to link nodes to themselves)
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
    
    # Loop to allocate vertices to nodes
    for i in range(Nhalf+1,N+1): # Nodes from N/2+1,...,N belongs to B ...
        for j in range(1,Nhalf+1): # ...and nodes from 0,1,...N/2+1 belongs to A.  
            pBA = np.random.uniform(0,1) #get a random draw bw 0 and 1
            if pBA>(1-c): #if above the probality that a node goes from 
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
        for j in range(Nhalf+1,N+1): #... and nodes from 0,1,...,N/2+1 belongs to A
            pBB = np.random.uniform(0,1) 
            if (j!=i) and pBB>(1-d): # additional conditions is i different from j (do not want to link nodes to themselves)
                G.add_edges_from([(i,j)]) #add a directed edge from i to j
           
                
    # Draw the graph created
    setA = list(range(1,Nhalf+1))
    pos=nx.spring_layout(G) # Draw the directed edges
    nx.draw_networkx(G,pos=nx.bipartite_layout(G, setA)) #Draw the directed edges and nodes
    labels = nx.get_edge_attributes(G,'weight') #create the labels (name of the nodes)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels) #add the label to the plot
    filename = "stoch_bloc.png"
    plt.savefig(filename, bbox_inches='tight')

    return(G)


## ------- Create the graph -------
np.random.seed(123)
G = stochastic_block_model(10)

## ------- Resolve using greedy algorithm -------
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}'.format('Set 0','Set 1','Cut Size'))
print('-' * 60)
print(greedy_algorithm(G))

## ------- Set up our QUBO dictionary -------
Q = defaultdict(int)

# # Update Q matrix for every edge in the graph
for i, j in G.edges:
    Q[(i,i)]+= -1
    Q[(j,j)]+= -1
    Q[(i,j)]+= 2


# # ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 8
numruns = 10

# # Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')

# # ------- Print results to user -------
print('-' * 60)
print('{:>15s}{:>15s}{:^15s}{:^15s}'.format('Set 0','Set 1','Energy','Cut Size'))
print('-' * 60)
for sample, E in response.data(fields=['sample','energy']):
    S0 = [k for k,v in sample.items() if v == 0]
    S1 = [k for k,v in sample.items() if v == 1]
    print('{:>15s}{:>15s}{:^15s}{:^15s}'.format(str(S0),str(S1),str(E),str(int(-1*E))))

# # ------- Display results to user -------
# # Grab best result
# # Note: "best" result is the result with the lowest energy
# # Note2: the look up table (lut) is a dictionary, where the key is the node index
# #   and the value is the set label. For example, lut[5] = 1, indicates that
# #   node 5 is in set 1 (S1).
# lut = response.first.sample

# # Interpret best result in terms of nodes and edges
# S0 = [node for node in G.nodes if not lut[node]]
# S1 = [node for node in G.nodes if lut[node]]
# cut_edges = [(u, v) for u, v in G.edges if lut[u]!=lut[v]]
# uncut_edges = [(u, v) for u, v in G.edges if lut[u]==lut[v]]

# # Display best result
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, nodelist=S0, node_color='r')
# nx.draw_networkx_nodes(G, pos, nodelist=S1, node_color='c')
# nx.draw_networkx_edges(G, pos, edgelist=cut_edges, style='dashdot', alpha=0.5, width=3)
# nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, style='solid', width=3)
# nx.draw_networkx_labels(G, pos)

# filename = "maxcut_plot.png"
# plt.savefig(filename, bbox_inches='tight')
# print("\nYour plot is saved to {}".format(filename))
