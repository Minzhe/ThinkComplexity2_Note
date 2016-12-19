import networkx as nx

G = nx.DiGraph()

G.add_node('Alice')
G.add_node('Bob')
G.add_node('Chuck')

G.nodes()

G.add_edge('Alice', 'Bob')
G.add_edge('Alice', 'Chuck')
G.add_edge('Bob', 'Alice')
G.add_edge('Bob', 'Chuck')

nx.draw_circular(G, node_size=2000, with_lables=True)

pos = dict(Albany=(-74, 43), Boston=(-71, 42), NYC=(-74, 41), Philly=(-75, 40))
G = nx.Graph()
G.add_nodes_from(pos)
drive_times = {('Albany', 'Boston'): 3, ('Albany', 'NYC'): 4, ('Boston', 'NYC'): 4, ('NYC', 'Philly'): 2}
G.add_edges_from(drive_times)
nx.draw(G, pos, node_shape='s', node_size=2500, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=drive_times)
