from math import *  # useful math functions
import numpy as np  # useful array objects
# (also a core scientific computing library)
import matplotlib.pyplot as plt  # nice plotting commands
from random import random, randint, choice  # random number generation functions


# Part A, midpoint function
def midpoint(t1, t2, fraction):
    return (t1[0] + t2[0]) * fraction, (t1[1] + t2[1]) * fraction


# Part B, setup
# corners = [(0, 0), (1, 0), (0.5, sqrt(3) / 2)]  # 3 corners

def get_corner_coords(num_sides):
    # Parameters for the polygon
    center_x, center_y = 0.5, 0.5
    radius = 0.5

    # Calculate the coordinates
    polygon_coords = []
    for i in range(num_sides):
        angle = radians(360 / num_sides * i + 90)  # Convert angle to radians
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        polygon_coords.append((x, y))

    return polygon_coords


corners = get_corner_coords(3)
fraction = 1 / 2

N = 1000
x = np.zeros(N)
y = np.zeros(N)

no_repeat_vertice = True

# Part C, starting position
x[0] = random()
y[0] = random()

# Part D, loops
if not no_repeat_vertice:
    for i in range(N):
        if i == 0:
            continue
        # choose a random point from the corners list
        corner = choice(corners)
        # Calculate the midpoint and
        # save to the point list
        x[i], y[i] = midpoint((x[i - 1], y[i - 1]), corner, fraction)
else:  # change rule
    for i in range(N):
        if i == 0:
            corner = choice(corners)
            continue
        # choose a random point apart from the last node
        corner = choice([i for i in corners if i != corner])
        # calculate the midpoint
        x[i], y[i] = midpoint((x[i - 1], y[i - 1]), corner, fraction)

# Part E, plotting the result
plt.figure()
plt.scatter(x, y)
xcoords, ycoords = zip(*corners)
plt.scatter(xcoords, ycoords, color='red')  # plot the triangle vertices
plt.show()
