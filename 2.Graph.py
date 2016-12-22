from __future__ import print_function, division

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import networkx as nx
import numpy as np

# colors from our friends at http://colorbrewer2.org
COLORS = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462',
          '#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']


### 2.1 Graph
#######################
G = nx.DiGraph()

G.add_node('Alice')
G.add_node('Bob')
G.add_node('Chuck')

G.nodes()

G.add_edge('Alice', 'Bob')
G.add_edge('Alice', 'Chuck')
G.add_edge('Bob', 'Alice')
G.add_edge('Bob', 'Chuck')

nx.draw_circular(G, 
                 node_color=COLORS[0], 
                 node_size=2000, 
                 with_labels=True)
plt.axis('equal')
# plt.savefig('chap02-1.pdf')

pos = dict(Albany=(-74, 43), Boston=(-71, 42), NYC=(-74, 41), Philly=(-75, 40))
G = nx.Graph()
G.add_nodes_from(pos)
drive_times = {('Albany', 'Boston'): 3, ('Albany', 'NYC'): 4, ('Boston', 'NYC'): 4, ('NYC', 'Philly'): 2}
G.add_edges_from(drive_times)
nx.draw(G, pos, 
        node_color=COLORS[1], 
        node_shape='s', 
        node_size=2500, 
        with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=drive_times)

### 2.2 completeGraph
##########################

def all_pairs(nodes):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i > j:
                yield u, v

def make_complete_graph(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(all_pairs(nodes))
    return(G)

complete = make_complete_graph(10)
nx.draw_circular(complete, node_size=1000, with_labels=True)

# for i, u in enumerate(range(10)):
#     print(i, u)

### 2.3 connectedGraph
##########################
complete.neighbors(0)

def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return(seen)

reachable_nodes(complete, 0)

def is_connected(G):
    start = next(G.nodes_iter())
    reachable = reachable_nodes(G, start)
    return(len(reachable) == len(G))

is_connected(complete)

### 2.4 randomGraph
#########################

from numpy.random import random

def flip(p):
    return random() < p

def random_pairs(nodes, p):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and flip(p):
                yield u, v

def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    G.add_edges_from(random_pairs(nodes, p))
    return G

random_graph = make_random_graph(10, 0.3)
len(random_graph.edges())

nx.draw_circular(random_graph, 
                 node_color=COLORS[3], 
                 node_size=1000, 
                 with_labels=True)


### 2.5 connectivity
########################

def reachable_nodes(G, start):
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(G.neighbors(node))
    return seen

def is_connected(G):
    start = next(G.nodes_iter())
    reachable = reachable_nodes(G, start)
    return len(reachable) == len(G)

reachable_nodes(complete, 0)
reachable_nodes(random_graph, 0)

is_connected(complete)
random_graph = make_random_graph(10, 0.1)
len(random_graph.edges())
is_connected(random_graph)

### 2.6 Probability of connectivity
####################################
def prob_connected(n, p, iters=100):
    count = 0
    for i in range(iters):
        random_graph = make_random_graph(n, p)
        if is_connected(random_graph):
            count += 1
    return count/iters

n = 10
prob_connected(n, 0.3, iters=10000)

pstar = np.log(n) / n
pstar

ps = np.logspace(-1.3, 0, 11)
ps

ys = [prob_connected(n, p, 1000) for p in ps]

for p, y in zip(ps, ys):
    print(p, y)

# import thinkplot

# thinkplot.vlines([pstar], 0, 1, color='gray')
# thinkplot.plot(ps, ys)
# thinkplot.config(xlabel='p', ylabel='prob connected', xscale='log', xlim=[ps[0], ps[-1]])

