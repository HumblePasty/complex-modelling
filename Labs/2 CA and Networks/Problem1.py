import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import linalg as la
import math
import scipy
from matplotlib.colors import ListedColormap

import random

print(scipy.__version__)
print(nx.__version__)

neighrule = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 0,
    (0, 1, 1): 1,
    (1, 0, 0): 1,
    (1, 0, 1): 1,
    (1, 1, 0): 1,
    (1, 1, 1): 0
}

initialcond = [0, 0, 0, 0, 1, 0, 0, 0]  # for testing

L = len(
    initialcond)  # 9 #10 #14 #50  # Grid size (start with something small - eventually running something relatively "big" like size 14 on rule 110 and rule 30 is fun)


# Takes a configuration and returns the corresponding integer
def config2int(config):
    return int(''.join(map(str, config)),
               2)  # maps the config->strings, joins them, and then converts to int from binary


# Takes an integer and converts it to a configuration (list of cell states)
def int2config(x):
    return [1 if x & 2 ** i > 0 else 0 for i in range(L - 1, -1, -1)]


def update(config):
    nextconfig = [0] * L
    for x in range(L):
        nextconfig[x] = neighrule[(config[(x - 1) % L], config[x], config[(x + 1) % L])]
    return nextconfig


# Run the model for a few steps and plot
steps = 50
output = np.zeros([steps, L])
# output[0,:] = int2config(2)
output[0, :] = initialcond
for i in range(1, steps):
    output[i, :] = update(output[i - 1, :])
plt.cla()
cmap = ListedColormap([(0, 39 / 255, 76 / 255), (241 / 255, 196 / 255, 0)])  # for fun use maize & blue colors
plt.imshow(output, cmap=cmap)
plt.show()

g = nx.DiGraph()  # Make an empty graph that will be the phase space

for x in range(2 ** L):
    g.add_edge(x, config2int(update(int2config(x))))

print(g)

# Plot each connected component of the phase space
ccs = [cc for cc in nx.connected_components(g.to_undirected())]
n = len(ccs)
print(n)
w = math.ceil(math.sqrt(n))
h = math.ceil(n / w)

plt.figure(1, figsize=(20, 20))
for i in range(n):
    plt.subplot(h, w, i + 1)
    # nx.draw_networkx(nx.subgraph(g, ccs[i]), with_labels=True)
    subg = nx.subgraph(g, ccs[i])
    attr = set().union(*nx.attracting_components(subg))
    attr_labels = {x: x for x in attr}
    other_labels = {x: x for x in (set(subg.nodes()) - attr)}
    pos = nx.spring_layout(subg)
    nx.draw_networkx_labels(subg, pos, labels=attr_labels, font_color='black')
    nx.draw_networkx_labels(subg, pos, labels=other_labels, font_color='white')
    nx.draw_networkx_nodes(subg, pos, nodelist=(set(subg.nodes()) - attr), node_color='#2F65A7',
                           node_size=300,
                           alpha=0.8)
    nx.draw_networkx_nodes(subg, pos, nodelist=attr, node_color='#FFCB05', node_size=300, alpha=0.8, label='Attractor')
    nx.draw_networkx_edges(subg, pos, width=1.0, alpha=0.8)

plt.show()

steps = 20

# explore the basins
plt.figure(1, figsize=(20, 20))
for i in range(n):
    plt.subplot(h, w, i + 1)
    output = np.zeros([steps, L])
    # select a random initial condition from the connected component
    rnd_config = random.choices(list(ccs[i]), k=1)[0]
    startconfig = int2config(rnd_config)
    # add title to the subplots
    plt.title(
        f'Initial condition for component {i + 1}: \n' + f'Config {rnd_config}: ' + ''.join(map(str, startconfig)))
    output[0, :] = startconfig
    for i in range(1, steps):
        output[i, :] = update(output[i - 1, :])
    cmap = ListedColormap([(0, 39 / 255, 76 / 255), (241 / 255, 196 / 255, 0)])  # for fun use maize & blue colors
    plt.imshow(output, cmap=cmap)

plt.show()

# another plot that color the nodes based on the centrality
plt.figure(1, figsize=(20, 20))
plt.title('Plot with nodes colored based on closeness centrality')
for i in range(n):
    plt.subplot(h, w, i + 1)
    subg = nx.subgraph(g, ccs[i])
    attr = set().union(*nx.attracting_components(subg))
    attr_labels = {x: x for x in attr}
    other_labels = {x: x for x in (set(subg.nodes()) - attr)}
    pos = nx.spring_layout(subg)
    # calculate the closeness centrality
    centrality = nx.closeness_centrality(subg)
    # color the nodes based on the centrality
    nx.draw_networkx_labels(subg, pos, labels=attr_labels, font_color='yellow')
    nx.draw_networkx_labels(subg, pos, labels=other_labels, font_color='black')
    nx.draw_networkx_nodes(subg, pos, node_color=list(centrality.values()),
                           cmap=plt.cm.Reds, node_size=300,
                           alpha=0.8)
    nx.draw_networkx_edges(subg, pos, width=1.0, alpha=0.8)

plt.show()

plt.figure(1, figsize=(20, 20))
plt.title('Plot with nodes colored based on betweenness centrality')
for i in range(n):
    plt.subplot(h, w, i + 1)
    subg = nx.subgraph(g, ccs[i])
    attr = set().union(*nx.attracting_components(subg))
    attr_labels = {x: x for x in attr}
    other_labels = {x: x for x in (set(subg.nodes()) - attr)}
    pos = nx.spring_layout(subg)
    # calculate the closeness centrality
    centrality = nx.betweenness_centrality(subg)
    # color the nodes based on the centrality
    nx.draw_networkx_labels(subg, pos, labels=attr_labels, font_color='yellow')
    nx.draw_networkx_labels(subg, pos, labels=other_labels, font_color='black')
    nx.draw_networkx_nodes(subg, pos, node_color=list(centrality.values()),
                           cmap=plt.cm.Reds, node_size=300,
                           alpha=0.8)
    nx.draw_networkx_edges(subg, pos, width=1.0, alpha=0.8)

plt.show()