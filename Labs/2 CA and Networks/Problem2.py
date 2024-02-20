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
