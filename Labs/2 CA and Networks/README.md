# Lab 2: Cellular Automata & Intro to Networks

> Author: Haolin Li (haolinli@umich.edu)
>
> Last Updated: 2/19/2024
>
> Instruction Link: https://epimath.org/cscs-530-materials/Labs/Lab2.html
>
> Course Repository: https://github.com/humblepasty/complex-modelling



## Problem 1: One-dimensional cellular automata

### Problem 1A

This is rule 122.

The binary rule set for this:

```
0 1 1 1 1 0 1 0
```



### Problem 1B

Trying with the starting config of [0, 0, 0, 0, 1, 0, 0, 0] for 50 steps

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/InitialConfig.png" style="zoom:50%;" />

The phase spaces plot with highlighted attracting components and labels:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/PhaseSpace.png" style="zoom:33%;" />

As the plot show, there are `8` basins of attraction in this situation

We can see from the plot that:

- Basin 1 will lead to a constant state (configuration 0) where all cells are 0
- Basin 2 will lead to a oscillation with a period of 2 from config 170 to config 85. 
- Basin 3 will lead to a oscillation with a period of 6.
- Basin 4 will lead to a oscillation with a period of 6, this basin is similar to basin 3.
- Basin 5 and 6 will also lead to a oscillation with a period of 6, similar with 3 and 4.
- Basin 7 and 8 will lead to a oscillation with a period of 2.



### Problem 1C

For each basin, choose a `random` starting configuration from the component, here is the plot:

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/CcsStatus.png)

### Problem 1D

Plot using `closeness centrality`

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/PhaseSpaceCentCol.png" style="zoom: 33%;" />

Plot using `betweenness centrality`

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/PhaseSpaceBwtCentCol.png" style="zoom:33%;" />

Nodes with high centrality usually are attracting sub-component nodes.



### Code Used in Problem 1

```python
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
```



## Problem 2

### Problem 2A

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/langtons_ant.gif)

***Note: The pdf does not show animated picture. See animated picture [here](https://drive.google.com/file/d/1tyKGC-OcZ90xF7fz7SBaop6A3WPbY6Jo/view?usp=sharing)***



### Problem 2B

Ways to place the ant:
$$
4\times 11\times 11
$$
Configurations of the cells:
$$
2^{11\times 11}
$$
Thus, total number of initial states:
$$
4\times 11\times 11\times 2^{11\times 11}
$$
Given the enormity of the total number of initial states, creating a comprehensive map or network of all these states and their transitions is not practically feasible with current computational resources. The complexity and size of the phase space make it implausible to fully characterize or analyse the system's phase space network exhaustively.



### Problem 2C

Animated picture with 1000 steps:

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/langtons_ant2.gif)

***Note: The pdf does not show animated picture. See animated picture [here](https://drive.google.com/file/d/1wYO-yNb38KcA0kbTfYarOGKscl0aywuG/view?usp=sharing)***

Final Configuration after 10500 steps

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/2C10500steps.png)

The ant dynamics:

1. **Initial Phase:** The ant starts with a seemingly random walk, creating complex patterns as it turns and flips the color of cells.
2. **Emergence of Order:** After a variable number of steps, which can depend on the initial configuration, the ant begins to generate a recurring pattern.
3. **Highway Phase:** Eventually, the ant enters a stable cycle, creating a diagonal "highway" pattern that it follows indefinitely, moving away from the initial area in a straight line.



### Problem 2D

**Simulation result with 100 sample runs of 10500 steps**

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/ant100samples.png)

***See large picture [here](https://drive.google.com/file/d/1cRS6ukqjQIm73CGggXfxkWjgIfakfEXm/view?usp=sharing)***

- It seems like the majority of runs will diverge into the “highway” phase. We can only find few samples in the picture that remains in the random walk phase.
- The fraction of runs converging to a highway within 10500 steps is about 0.7 (roughly)



### Code Used in Problem 2

```python
from matplotlib import pyplot as plt
from pylab import *
from matplotlib.animation import FuncAnimation
import numpy as np

n = 11  # size of space: n x n
step = 200  # step counter


def initialize():
    global config, ant  # Things we need to access from different functions go here (discuss globals)

    # Build our grid of agents - fill with zeros for now
    config = zeros([n, n])

    # Set the initial configuration of the langton's ant
    # [x, y, direction] where direction is 0, 1, 2, 3 for up, right, down, left
    ant = [n // 2, n // 2, 0]  # start in the middle facing up


def update():
    global config, ant

    # Update the configuration of the langton's ant
    if config[ant[0], ant[1]] == 1:
        ant[2] = (ant[2] + 1) % 4
        config[ant[0], ant[1]] = 0
    else:
        ant[2] = (ant[2] - 1) % 4
        config[ant[0], ant[1]] = 1

    # Move the ant
    ant[0] = (ant[0] + [0, 1, 0, -1][ant[2]]) % n  # wrap around the edges
    ant[1] = (ant[1] + [1, 0, -1, 0][ant[2]]) % n


antmarker = {0: '^', 1: '>', 2: 'v', 3: '<'}

fig, ax = plt.subplots()


def animate(i):
    global config, ant
    update()
    ax.clear()  # Clear the previous image before drawing the next one
    ax.imshow(config, cmap='binary')
    ax.scatter(ant[1], ant[0], s=100, c='red', marker=antmarker[ant[2]])
    ax.set_title(f"Step: {i}")

initialize()

ani = FuncAnimation(fig, animate, frames=step, repeat=False)
ani.save('langtons_ant.gif', fps=10)


# new configuration
n = 50  # size of space: n x n
step = 1000  # step counter


def initialize2C():
    global config, ant  # Things we need to access from different functions go here (discuss globals)

    # Build our grid of agents - fill with zeros for now
    config = zeros([50, 50])

    # set the center 3x3 randomly
    config[23:26, 23:26] = np.random.randint(0, 2, (3, 3))

    # Set the initial configuration of the langton's ant
    # [x, y, direction] where direction is 0, 1, 2, 3 for up, right, down, left
    ant = [n // 2, n // 2, 0]  # start in the middle facing up


initialize2C()
for j in range(10500):
    update()
plt.imshow(config, cmap='binary')
plt.title("Final Configuration of Problem 2C with 1000 steps")
plt.show()

fig, ax = plt.subplots()

ani2 = FuncAnimation(fig, animate, frames=step, repeat=False)
ani2.save('langtons_ant2.gif', fps=10)

# Problem 2d
n = 50  # size of space: n x n
step = 10500  # step counter

# show the final configuration of 100 samples
# create a 10x10 grid of subplots
subplots(10, 10, figsize=(50, 50))

for i in range(100):
    initialize2C()
    for j in range(step):
        update()
    subplot(10, 10, i + 1)
    imshow(config, cmap='binary')
    title(f"Sample {i + 1}")
```



## Problem 3

The network

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookNetwork.png" style="zoom:50%;" />

### Problem 3A

The bar plot of distribution of the degree of polbook:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookDegreeDistri.png" style="zoom:50%;" />

In log-log plot:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookDegreeDistri_log.png" style="zoom: 50%;" />

The degree distribution of this network does not match the distribution of the scale-free network, which should be:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/power_law_degree_distribution_scatter.png" style="zoom: 67%;" />

Thus this is not expected to be a scale-free network



### Problem 3B

Network Diagram with nodes coloured based on closeness centrality

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookNetwork_closeness.png" style="zoom:50%;" />

- In network analysis, nodes with the highest closeness centrality are those that, on average, are shortest in distance to all other nodes in the network. This centrality measure identifies nodes that can efficiently spread information or navigate the network due to their minimal path lengths to others. It highlights nodes that are strategically positioned to influence the network by quickly reaching or being reached by all other nodes.
- Closeness centrality captures a node's accessibility to the rest of the network, reflecting its potential for influence and visibility. It is crucial for understanding which nodes are central in terms of information dissemination, epidemic spreading, and ensuring network connectivity. High closeness centrality points to nodes that are vital for the network's functionality, emphasizing their role in facilitating or controlling the flow within the network.



### Problem 3C

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookNetwork_cata.png)

```python
print(nx.attribute_assortativity_coefficient(g, 'value'))
```

```
0.7233077584970603
```



- An assortativity coefficient of 0.7233077584970603 is relatively high, indicating a strong tendency for nodes (in this case, books) to be connected to other nodes with similar attributes, here referring to their political leanings (liberal, conservative, or neutral). In the context of a network of books, this suggests that books are predominantly linked (e.g., co-purchased, recommended together, or thematically similar) to other books with the same political orientation.

- This strong assortativity by political leaning reveals significant homophily in the book-purchasing habits of Amazon customers. It implies that customers who buy books with a particular political leaning (liberal, conservative, or neutral) are likely to buy or interact with other books of the same political orientation. This could be reflective of broader social and informational silos or echo chambers, where individuals tend to seek out and consume information that aligns with their existing beliefs and values. For retailers like Amazon, this might influence how recommendations are generated and could highlight the importance of understanding customer preferences not just in terms of genres or authors, but also in terms of the political content or orientation of books.



### Problem 3D

![](https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/polbookNetwork_commu.png)

The communities align with the liberal/conservative/neutral labels on the nodes quite well. Political books in the same category tend to stay in the same community.



### Code Used in Problem 3

```python
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
```



## Problem 4

### Problem 4A

#### Topic 1: Using ABM to Model the Traffic Flow

- Interested in understanding the dynamics of urban traffic flows, especially how individual vehicle movements contribute to overall traffic patterns, congestion, and pollution in urban settings.
- The goal would be to identify strategies (or at least identify possibilities) for congestion mitigation and improving urban mobility.
- ABMs can simulate the behaviours of individual drivers and vehicles under different traffic rules, urban layouts, and policy interventions, offering insights into complex interactions that lead to traffic congestion and potential solutions for urban transportation planning.



#### Topic 2: Using Cellular Automaton to Predict Landcover Change (in aligned with another friend’s interest)

- Using Cellular Automata (CA) to predict land cover change models how land use evolves due to human and environmental influences.
- The goal is to forecast changes under scenarios like urban growth or conservation strategies. 
- ABMs enhance this by incorporating individual behaviours (e.g., urban expansion, farming) and natural processes, offering nuanced insights into landscape evolution. This approach helps in planning and assessing the impact of land management policies on land cover dynamics.



### Problem 4B

#### For Topic 1:

- The sample project that first caught by attention into this topic (mentioned in class)

  https://github.com/HumblePasty/traffic-simulation-de

- Bazghandi, Ali. “Techniques, Advantages and Problems of Agent Based Modeling for Traffic Simulation.” (2012).



#### For Topic 2:

- Saputra, Muhammad Hadi and Han Soo Lee. “Prediction of Land Use and Land Cover Changes for North Sumatra, Indonesia, Using an Artificial-Neural-Network-Based Cellular Automaton.” *Sustainability* (2019): n. pag. https://doi.org/10.3390/SU11113024
- Kamusoko, Courage et al. “Rural sustainability under threat in Zimbabwe - Simulation of future land use/cover changes in the Bindura district based on the Markov-cellular automata model.” *Applied Geography* 29 (2009): 435-447. https://doi.org/10.1016/J.APGEOG.2008.10.002



### Problem 4C

#### For Topic 1:

1. **Agents in the Model:** Vehicles, pedestrians, and possibly cyclists, each with their own behaviors, goals, and constraints. 
2. **Interactions to Consider:** Agents would interact primarily on a network, representing the urban road system. Interactions include following traffic signals, yielding, merging into traffic, and avoiding collisions with other vehicles and pedestrians. Implementation can use a combination of network-based interactions (where agents move on predefined paths that represent roads) and grid or continuous space models for areas like intersections or pedestrian zones. Agent decisions can be modeled using rule-based systems or more complex decision-making algorithms that consider the state of neighboring agents and traffic controls.
3. **Type of Environment:** The environment would be a simulated urban area, possibly based on real-world data to accurately reflect street layouts, traffic signal placements, and other infrastructure elements like parks, buildings, and pedestrian zones. This could be implemented using Geographic Information Systems (GIS) data to create a realistic simulation environment.
4. **Existing Models or Frameworks:** Several ABM frameworks could be adapted for this purpose, such as MATSim, SUMO (Simulation of Urban MObility), or Agent-Based Simulation frameworks like NetLogo with extensions for road networks. These platforms offer tools for simulating traffic flows and can be customized to incorporate specific behaviors, policies, and infrastructure changes.
5. **Outcomes to Track:** Key outcomes could include metrics on traffic flow efficiency (e.g., average travel time, speed), congestion levels at different times and locations, environmental impacts (e.g., emissions), and the effectiveness of various mitigation strategies (e.g., changes to traffic signal timings, the introduction of congestion charges, or the promotion of alternative transportation modes). These outcomes will help in understanding the dynamics of urban traffic, identifying bottlenecks, and evaluating the potential benefits of different congestion mitigation strategies.



#### For Topic 2:

1. **Agents:** Represent different land uses (urban, agricultural, forest) and possibly human activities (farming, development).
2. **Interactions:** Agents interact with their immediate grid neighbors, with rules governing changes like urban expansion or deforestation based on neighboring cells' states.
3. **Environment:** A gridded landscape based on real-world data, where each cell has specific landcover attributes.
4. **Existing Models:** Can use frameworks like TerraME or NetLogo for creating adaptable CA models.
5. **Outcomes:** Focus on tracking landcover change rates, impacts of policies, and effects on biodiversity or resources. This streamlined approach aims to simulate and assess the dynamics of land use changes under various scenarios.