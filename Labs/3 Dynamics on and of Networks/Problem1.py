import matplotlib.pyplot as plt
from pylab import *
from IPython import display
import networkx as nx
import numpy as np
from copy import copy, deepcopy
import time

N = 100  # 100 #200  #1000

p_e = 0.06  # probability of edge generation (connection between two people)
p_i = 0.1  # 0.5 # infection probability per contact
p_r = 0.1  # recovery probability
beta = 0.8  # set to 0.5 for now

R0 = (N - 1) * p_e * p_i / p_r
# print(R0)

prev = []  # prevalence (total infected nodes/total nodes) in the network model
prev_mf = []  # prevalence (total infected nodes/total nodes) in the mean field model
susc_mf = []  # fraction of nodes that are susceptible in the mean field model


def initialize():
    global g, nextg, prev, prev_mf, susc_mf

    # Initialize network model
    prev = []
    prev_mf = []
    susc_mf = []

    # generate graph using watts_strogatz_graph
    g = nx.watts_strogatz_graph(N, round((p_e * (N - 1))), beta)  # start with a WS graph
    # g = nx.erdos_renyi_graph(N, p_e)  # start with an ER graph
    g.pos = nx.circular_layout(g)  # set the positions of the nodes (for plotting
    nx.set_node_attributes(g, 0, 'state')  # everyone starts off susceptible
    g.nodes[1]['state'] = 1  # set one node to be infected (index case)
    g.nodes[2]['state'] = 1
    g.nodes[3]['state'] = 1
    g.nodes[4]['state'] = 1
    nextg = g.copy()
    nextg.pos = g.pos
    prev.append(4 / len(g.nodes))  # initial prevalence in the real graph

    # Initialize mean field model
    susc_mf.append((N - 4) / N)  # initial susceptible fraction in the mean field model
    prev_mf.append(4 / N)  # initial prevalence in the mean field model


def update():
    global g, nextg, prev, prev_mf, susc_mf

    # Update network model
    curprev = 0
    nextg = g.copy()
    nextg.pos = g.pos
    for a in g.nodes:
        if g.nodes[a]['state'] == 0:  # if susceptible
            nextg.nodes[a]['state'] = 0
            for b in g.neighbors(a):
                if g.nodes[b]['state'] == 1:  # if neighbor b is infected
                    if random() < p_i:
                        nextg.nodes[a]['state'] = 1
        elif g.nodes[a]['state'] == 1:  # if infected
            curprev += 1
            nextg.nodes[a]['state'] = 2 if random() < p_r else 1
    prev.append(curprev / len(g.nodes()))  # prevalence in the real graph
    g = nextg.copy()
    g.pos = nextg.pos

    # Update mean field model
    susc_mf.append(susc_mf[-1] - (N - 1) * p_e * p_i * prev_mf[-1] * susc_mf[-1])
    prev_mf.append(prev_mf[-1] + (N - 1) * p_e * p_i * prev_mf[-1] * susc_mf[-1] - p_r * prev_mf[-1])


def observe():  # visualize the network
    global g, prev, prev_mf, susc_mf
    cla()
    nx.draw(g, cmap=cm.plasma, vmin=0, vmax=2,
            node_color=[g.nodes[i]['state'] for i in g.nodes],
            pos=g.pos)


# initialize()
# update()
# observe()
# plt.show()
#
# prev = []  # prevalence (total infected nodes/total nodes) in the network model
# prev_mf = []  # prevalence (total infected nodes/total nodes) in the mean field model
# susc_mf = []  # fraction of nodes that are susceptible in the mean field model
#
# initialize()
# for i in range(25):
#     update()
#     observe()
#     display.clear_output(wait=True)
#     display.display(gcf())
#     time.sleep(0.5)
#
# # plot the prevalence over time
# clf()
# epicurve = scatter(range(len(prev)), prev)
# scatter(range(len(prev_mf)), prev_mf)
# xlabel("Time")
# ylabel("Prevalence")
# # add a legend
# legend(["WS Network Model Simulation", "Mean Field Model Prediction"])
# plt.show()


# simulate the process with different beta values
beta_values = np.linspace(0, 1, 6)  # the bate values to simulate


def beta_comparison(beta_values):
    global beta, prev, prev_mf, susc_mf
    plt.figure(figsize=(15, 10))
    for index, b in enumerate(beta_values):
        beta = b
        initialize()

        for i in range(25):
            update()
            # observe()
            # display.clear_output(wait=True)
            # display.display(gcf())
            # time.sleep(0.5)
        subplot(2, 3, index + 1)
        epicurve = scatter(range(len(prev)), prev)
        scatter(range(len(prev_mf)), prev_mf)
        xlabel("Time")
        ylabel("Prevalence")
        title("Beta: " + str(round(b, 1)))
        legend(["WS Network Model Simulation", "Mean Field Model Prediction"])

    # plt.tight_layout()
    plt.suptitle("Network Model and Mean Field Model for Different Beta Values")
    plt.show()


# simulate the process with different beta values
# beta_comparison(beta_values)

# problem 1c

beta_values = np.linspace(0, 1, 6)  # the bate values to simulate


def beta_longer_sim(beta_values):
    global beta, prev, prev_mf, susc_mf
    timesteps = 100
    numruns = 100
    plt.figure(figsize=(15, 10))
    for index, b in enumerate(beta_values):
        beta = b
        prevarray = np.zeros([timesteps, numruns])
        plt.subplot(2, 3, index + 1)
        plt.title("Beta: " + str(round(b, 1)))
        initialize()
        for i in range(0, numruns):
            initialize()
            for j in range(1, timesteps):
                update()
            prevarray[:, i] = prev
            scatter(range(timesteps), prev, s=5)
        plot(range(timesteps), prev_mf, 'k', linewidth=4)

    plt.suptitle("100 runs of 100 timesteps for different beta values")
    plt.tight_layout()
    plt.show()


# beta_longer_sim(beta_values)

# problem 1d
def beta_longer_sim_1d(beta_values):
    global beta, prev, prev_mf, susc_mf
    timesteps = 100
    numruns = 100
    plt.figure(figsize=(15, 10))
    for index, b in enumerate(beta_values):
        beta = b
        prevarray = np.zeros([timesteps, numruns])
        plt.subplot(2, 3, index + 1)
        plt.title("Beta: " + str(round(b, 1)))
        initialize()
        for i in range(0, numruns):
            initialize()
            for j in range(1, timesteps):
                update()
            prevarray[:, i] = prev

        # Calculate median, 5% and 95% quantiles for each timestep
        median = np.median(prevarray, axis=1)
        quantile_5 = np.quantile(prevarray, 0.05, axis=1)
        quantile_95 = np.quantile(prevarray, 0.95, axis=1)

        # Plotting
        time_range = range(timesteps)
        plt.plot(time_range, median, 'r-', label='Median')
        plt.fill_between(time_range, quantile_5, quantile_95, color='blue', alpha=0.2, label='5% - 95% quantile')
        plt.plot(time_range, prev_mf, 'k--', linewidth=2, label='Mean-field prediction')

        plt.legend()

    plt.suptitle("100 runs of 100 timesteps for different beta values, with quantiles and mean-field prediction")
    plt.tight_layout()
    plt.show()


# beta_longer_sim_1d(beta_values)

# problem 1e
timesteps = 100
numruns = 100
plt.figure(figsize=(5, 5))
prevarray = np.zeros([timesteps, numruns])
plt.title("Erdos-Renyi graph simulations with p_e=0.06")
prev = []
prev_mf = []
susc_mf = []

# generate graph using watts_strogatz_graph
g = nx.erdos_renyi_graph(N, 0.06)
g.pos = nx.circular_layout(g)  # set the positions of the nodes (for plotting
nx.set_node_attributes(g, 0, 'state')  # everyone starts off susceptible
g.nodes[1]['state'] = 1  # set one node to be infected (index case)
g.nodes[2]['state'] = 1
g.nodes[3]['state'] = 1
g.nodes[4]['state'] = 1
nextg = g.copy()
nextg.pos = g.pos
prev.append(4 / len(g.nodes))  # initial prevalence in the real graph

# Initialize mean field model
susc_mf.append((N - 4) / N)  # initial susceptible fraction in the mean field model
prev_mf.append(4 / N)  # initial prevalence in the mean field model

for i in range(0, numruns):
    initialize()
    for j in range(1, timesteps):
        update()
    prevarray[:, i] = prev

# Calculate median, 5% and 95% quantiles for each timestep
median = np.median(prevarray, axis=1)
quantile_5 = np.quantile(prevarray, 0.05, axis=1)
quantile_95 = np.quantile(prevarray, 0.95, axis=1)

# Plotting
time_range = range(timesteps)
plt.plot(time_range, median, 'r-', label='Median')
plt.fill_between(time_range, quantile_5, quantile_95, color='blue', alpha=0.2, label='5% - 95% quantile')
plt.plot(time_range, prev_mf, 'k--', linewidth=2, label='Mean-field prediction')

plt.legend()
plt.show()
