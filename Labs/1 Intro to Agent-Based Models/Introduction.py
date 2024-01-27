from pylab import *
from random import random
import sys
sys.path.append('../PyCX-master')
import pycxsimulator


n = 100
p = 0.5


def initialize():
    # Things we need to access from different functions go here (discuss globals)
    global config, nextconfig

    # Build our grid of agents - fill with zeros for now
    config = zeros([n, n])

    # Set them to vote yes with probability p
    for y in range(n):
        for x in range(n):
            if random() < p: config[x, y] = 1

    # Set the next timestep's grid to zeros for now (we'll update in the update function)
    nextconfig = zeros([n, n])


def update():
    global config, nextconfig

    # Go through each cell and check if they should change their vote in the next step
    for x in range(n):
        for y in range(n):
            count = 0  # variable to keep track of how many neighbors are voting yes

            for dx in [-1, 0, 1]:  # check the cell before/middle/after
                for dy in [-1, 0, 1]:  # check above/middle/below
                    # discuss nesting for loops vs. not---what does this change?

                    # Add to count if neighbor is voting yes (note you also count yourself!)
                    count += config[(x + dx) % n, (y + dy) % n]  # discuss

            # Now that we know how many neighbors are voting yes, decide what to do
            if config[x, y] == 0:  # if this agent was going to vote no
                nextconfig[x, y] = 1 if count > 4 else 0
                # note we only change the vote for nextconfig, not config!

            else:  # otherwise agent was going to vote yes (could also do elif)
                nextconfig[x, y] = 0 if (8 - (count - 1)) > 4 else 1
                # note we reduced count by 1 since count included self

    # advance config forward one step and reset nextconfig
    config, nextconfig = nextconfig, zeros([n, n])
    # Can also be a little more efficient and do config, nextconfig = nextconfig, config


def observe():
    global config, nextconfig
    cla()  # clear visualization
    imshow(config, vmin=0, vmax=1, cmap=cm.binary)  # display grid!

pycxsimulator.GUI().start(func=[initialize, observe, update])