import matplotlib.pyplot as plt
import pylab as plb
from IPython import display
import networkx as nx
import numpy as np
from copy import copy, deepcopy
import time

# generate the graph
N = 100
m = 2

pct_failed = []


def initialize(load_min, load_max):
    global g, pct_failed

    # generate graph using barabasi_albert_graph
    g = nx.barabasi_albert_graph(N, m)
    # g.pos = nx.spring_layout(g)  # set the positions of the nodes (for plotting
    g.pos = nx.spectral_layout(g)  # set the positions of the nodes (for plotting
    nextg = g.copy()
    nextg.pos = g.pos

    # capacity with normal distribution of mean 100, standard deviation 10
    nx.set_node_attributes(g, 0, 'capacity')
    for a in g.nodes:
        g.nodes[a]['capacity'] = np.random.normal(100, 10)

    # electicity load random from 0.2 to 1.0 of capacity
    nx.set_node_attributes(g, 0, 'load')
    for a in g.nodes:
        g.nodes[a]['load'] = np.random.uniform(load_min, load_max) * g.nodes[a]['capacity']

    # opreational status
    nx.set_node_attributes(g, 'running', 'status')

    # attributes for plot
    pct_failed = []
    pct_failed.append(0)


def update():
    global g, nextg, prev, prev_mf, susc_mf

    # Update network model
    nextg = g.copy()
    nextg.pos = g.pos
    for a in g.nodes:
        if g.nodes[a]['status'] == 'running':
            if g.nodes[a]['load'] > g.nodes[a]['capacity']:
                nextg.nodes[a]['status'] = 'failed'
                # evenly distribute the load to the neighbors
                for b in g.neighbors(a):
                    nextg.nodes[b]['load'] += g.nodes[a]['load'] / len(list(g.neighbors(a)))
            else:
                nextg.nodes[a]['status'] = 'running'
        elif g.nodes[a]['status'] == 'failed':
            nextg.nodes[a]['status'] = 'failed'
    g = nextg.copy()
    g.pos = nextg.pos

    pct_failed.append(len([n for n in g.nodes if g.nodes[n]['status'] == 'failed']) / len(g.nodes))


def observe():
    global g, nextg, prev, prev_mf, susc_mf

    # Plot the graph
    # plb.clf()
    # color the nodes according to their status
    color_map = []
    for node in g:
        if g.nodes[node]['status'] == 'running':
            color_map.append('green')
        else:
            color_map.append('red')
    # adjust the size of the node according to the capacity
    node_size = []
    for node in g:
        node_size.append(g.nodes[node]['capacity'])
    nx.draw(g, pos=g.pos, node_color=color_map, with_labels=True, node_size=node_size)
    # plb.show()


# problem 2a

# start the simulation with no failed nodes
# initialize(0.2, 1.0)
# for i in range(25):
#     update()
# #     observe()
# #     time.sleep(0.5)
# #     display.clear_output(wait=True)
#
# observe()
# plt.show()
#
# plb.plot(pct_failed)
# plb.xlabel('Timestep')
# plb.ylabel('Percentage of failed nodes')
# plb.title('Simulation with no failed nodes')
# plb.show()
#
#
# # start with one node over capacity
# initialize(0.2, 1.0)
#
# # an example of a specific node
# # g.nodes[2]['load'] = g.nodes[2]['capacity'] + 1
#
# # set a random node to hold more load than its capacity
# failed_node = np.random.choice(g.nodes, 1)
# g.nodes[failed_node[0]]['load'] = g.nodes[failed_node[0]]['capacity'] + 1
#
# timestep = 50
# for i in range(25):
#     update()
#     # observe()
#     # time.sleep(0.5)
#     # display.clear_output(wait=True)
#
# observe()
#
# # plot the percentage of failed nodes over time
# plb.plot(pct_failed)
# plb.xlabel('Timestep')
# plb.ylabel('Percentage of failed nodes')
# plb.title('Simulation with one node over capacity')
# plb.show()

# problem 2b

# different load distributions
# load_settings = [(0.2, 1.0), (0.5, 1.0), (0.8, 1.0), (0.2, 0.5), (0.2, 0.8), (0.2, 0.2)]
# # two 2 times 3 figures
# final_network = plt.figure(figsize=(15, 10))
# # title for each figure
# plt.suptitle('Final networks')
# cascade = plt.figure(figsize=(15, 10))
# plt.suptitle('Cascade Dynamics')
#
# for index, load_setting in enumerate(load_settings):
#     initialize(load_setting[0], load_setting[1])
#     failed_node = np.random.choice(g.nodes, 1)
#     g.nodes[failed_node[0]]['load'] = g.nodes[failed_node[0]]['capacity'] + 1
#
#     for i in range(50):
#         update()
#
#     plt.figure(final_network.number)
#     plt.subplot(2, 3, index + 1)
#     plt.title('Load setting: ' + str(load_setting))
#     observe()
#
#     plt.figure(cascade.number)
#     plt.subplot(2, 3, index + 1)
#     plt.plot(pct_failed, label='Load setting: ' + str(load_setting))
#     plt.legend()
#
# plt.show()

# problem 2c
def initialize_real(load_min, load_max):
    global g, pct_failed

    # generate graph using barabasi_albert_graph
    g = nx.read_gml('power/power.gml', label='id')
    g.pos = nx.spectral_layout(g)  # set the positions of the nodes (for plotting
    nextg = g.copy()
    nextg.pos = g.pos

    # capacity with normal distribution of mean 100, standard deviation 10
    nx.set_node_attributes(g, 0, 'capacity')
    for a in g.nodes:
        g.nodes[a]['capacity'] = np.random.normal(100, 10)

    # electicity load random from 0.2 to 1.0 of capacity
    nx.set_node_attributes(g, 0, 'load')
    for a in g.nodes:
        g.nodes[a]['load'] = np.random.uniform(load_min, load_max) * g.nodes[a]['capacity']

    # opreational status
    nx.set_node_attributes(g, 'running', 'status')

    # attributes for plot
    pct_failed = []
    pct_failed.append(0)

# load network from file
initialize_real(0.5, 1.0)

# plot the network
# plb.clf()
# nx.draw(g, pos=g.pos, node_size=10, node_color='blue')
# plb.title('Power Grid Network, Spectral Layout')
# plb.show()
#
# # plot the degree distribution on a log-log scale
# degrees = [g.degree(n) for n in g.nodes]
# degree_counts = np.unique(degrees, return_counts=True)
#
# # the probability of a node having a certain degree
# degree_probs = degree_counts[1] / N
#
# plt.figure()
# plt.loglog(degree_counts[0], degree_probs, 'b.', markersize=12)
#
# slope, intercept = np.polyfit(np.log(degree_counts[0]), np.log(degree_probs), 1)
# plt.loglog(degree_counts[0], np.exp(intercept) * degree_counts[0] ** slope, 'r-')
#
# plt.legend(['Degree distribution', 'Power-law fit'])
# plt.xlabel('Degree (# edges)')
# plt.ylabel('Fraction of nodes')
# plt.title('Degree Distribution on Log-Log Scale')
#
# plt.show()

# problem 2d
# run 20 simulations with 50 timesteps each
final_failed = []
for i in range(20):
    initialize_real(0.5, 1.0)
    failed_node = np.random.choice(g.nodes, 1)
    g.nodes[failed_node[0]]['load'] = g.nodes[failed_node[0]]['capacity'] + 1

    for j in range(50):
        update()

    # record the final percentage of failed nodes
    final_failed.append(pct_failed[-1])
    print('Simulation', i, 'done')

# plot the histogram of the final percentage of failed nodes
# print in tabular form
print('Final percentage of failed nodes')
print(final_failed)

bins = np.linspace(np.min(final_failed), np.max(final_failed), 11)  # 创建10个bins
plt.hist(final_failed, bins=bins)
plt.xlabel('Final percentage of failed nodes')
plt.ylabel('Frequency')
plt.title('Histogram of final percentage of failed nodes')
plt.show()



