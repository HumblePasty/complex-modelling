from math import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

g = nx.read_gml('polbooks/polbooks.gml')

pos=nx.spring_layout(g) # positions for all nodes
nx.draw_networkx(g, pos, with_labels = False, node_size = 100) # draw network
plt.show()

# Problem 3A
# Plot the degree distribution of the graph
# degree_sequence = [d for n, d in g.degree()]  # degree sequence
# plt.hist(degree_sequence, bins=20)
# plt.title("Degree Histogram")
# plt.ylabel("Count")
# plt.xlabel("Degree")
# plt.show()

# Another way to plot the degree distribution
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
dmax = max(degree_sequence)
y = np.bincount(degree_sequence)
x = np.arange(len(y))
plt.bar(x, y)
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()

# Plot on a log-log scale
plt.loglog(x, y, 'b.') # in-degree
plt.title("Degree Distribution (log-log scale)")
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.show()

# Problem 3B
# Plot the network with nodes colored by centrality
centrality = nx.closeness_centrality(g)
node_color = [centrality[n] for n in g.nodes()]
nx.draw_networkx(g, pos, with_labels = False, node_size = 100, node_color=node_color, cmap=plt.cm.Reds)
plt.show()

# Problem 3C
# Plot the network with nodes colored by book category
n,v = zip(*g.nodes(data=True))
# n = g.nodes(data=True)
node_color = []
for i in v:
    if i['value'] == 'l':
        node_color.append('blue')  # Blue for 'l'
    elif i['value'] == 'c':
        node_color.append('red')  # Red for 'c'
    else:
        node_color.append('grey')  # Grey for 'n' or any other condition

plt.title("Network with nodes colored by book category")
nx.draw_networkx(g, pos, with_labels = False, node_size = 100, node_color=node_color)
plt.show()

# calculate the assortativity of the graph
print(nx.attribute_assortativity_coefficient(g, 'value'))

# Problem 3D
# use greedy modularity optimization to find communities
from networkx.algorithms.community import greedy_modularity_communities
c = list(greedy_modularity_communities(g))
node_color_communities = []
for i in g.nodes():
    for j in range(len(c)):
        if i in c[j]:
            node_color_communities.append(j)
plt.title("Network with nodes colored by community")
nx.draw_networkx(g, pos, with_labels = False, node_size = 100, node_color=node_color_communities, cmap=plt.cm.tab20)
plt.show()

pass



