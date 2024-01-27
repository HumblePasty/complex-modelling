# CMPLXSYS - 530 Lab1 Writeup : Intro to Agent-Based Models

> Author: Haolin Li (haolinli@umich.edu)
>
> Last Updated: 1/26/2024
>
> Instruction Link: https://epimath.org/cscs-530-materials/Labs/Lab1.html
>
> Course Repository: https://github.com/humblepasty/complex-modelling



## Part 1 - Emoji-ABMs

### Problem 1

#### Problem 1A

**Code the model, supposing that each animal will move if less than 2 (25%) of its neighbours to be the same as it. Run the model a few times. What behaviour do you observe?**

http://tinyurl.com/haolin-schelling-emojimodel

> Rule set:
>
> Things with rules
>
> <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118144152900.png" alt="image-20240118144152900" style="zoom: 50%;" />
>
> The world
>
> <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153208311.png" alt="image-20240118153208311" style="zoom: 50%;" />

The cats and dogs started to gather into certain regions. Areas with a relatively larger percentage of cats/dogs tend to grow in size. Things stopped moving to new places after few iterations (stabilized quickly).



#### Problem 1B

**Adjust the number of neighbours required to be the same. How does the model behaviour vary as this number varies from 1 to 8?**

When set to 1

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118154822899.png" alt="image-20240118154822899" style="zoom:50%;" />

When set to 2

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153237577.png" alt="image-20240118153237577" style="zoom:50%;" />

When set to 3

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153259041.png" alt="image-20240118153259041" style="zoom:50%;" />

When set to 4

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153354390.png" alt="image-20240118153354390" style="zoom:50%;" />

When set to 5

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153516437.png" alt="image-20240118153516437" style="zoom:50%;" />

When set to 6 or above

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118153738582.png" alt="image-20240118153738582" style="zoom:50%;" />

Conclusion:

- When the number is set to 1-4:
  - The world will **stabilized** after a specific number of iterations. With increasing the % of required similarity (actually the number of  neighbouring cells required to be the same), **more iterations** are needed for the world to stabilize
  - **Fewer** number of **larger** groups are formed with higher number of value set
- When the number is set to be 5, only **two large groups** will be formed. The world will **not stabilized** with a number of  cells of cats and dogs wandering around on the blank spaces.
- When the number is set to higher than 5, the process seems to continue forever. The world seems to be chaotic in general. No obvious group will be formed and every animal cell is always trying to relocate them selves to other cells.



#### Problem 1C

**Now try out adjusting the balance between empty cells and animals. How does the model behaviour change as the density of whales and foxes increases or decreases? Do you always observe separation between the two types? Why do you think this is?**

What happens when increasing the # of blank cells:

> number of similarity required set: 6

Number of blank cells: low

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118155224704.png" alt="image-20240118155224704" style="zoom:50%;" />

Number of blank cells: medium

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118155327995.png" alt="image-20240118155327995" style="zoom:50%;" />

Number of blank cells: high

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118155433424.png" alt="image-20240118155433424" style="zoom:50%;" />

No. 

- If the amount of blank cell is set to 0, there will be no separation effect because no animal can relocate itself
- With a relatively small amount of blank cells, the world quickly stabilize slowly and the separation effect is not very obvious at start but will become obvious after a certain amount of iterations.
- As the number of blank cells increase, the stabilization process happens more quickly
- If the amount of blank cells exceed a certain level, the world will to be chaotic in general. No obvious group will be formed and every animal cell is always trying to relocate them selves to other cells



#### Problem 1D

**Lastly, let’s illustrate the importance of being careful about how we specify our rules. Re-run problems 1A - 1C with a slightly different rule: *each type of animal wants no more than Y of its neighbours to be the opposite type*. In other words, if there are more than 6 (75%) neighbours of the other type, it will move. Does this change the behaviour of the model? If so, how does the behaviour change? Why do you think this is?**

> Rule set:
>
> <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118160811767.png" alt="image-20240118160811767" style="zoom:50%;" />

Stabilized result:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240118162210384.png" alt="image-20240118162210384" style="zoom:50%;" />

The behaviour of the model changed if the rule is set in this way. The model seems to stabilized quickly. This is because few cell will have more than 6 neighbouring animals of the opposite kind. So there will be limited number of animals who want to relocate themselves.



### Problem 2: Build your own model!

*Start with a blank model and build a model of your choosing! In your write-up, document the model you built, including:*

- *The overall process your are modeling and/or question you are trying to address*
- *Agents: the types of agents in your model*
- *Rules/interactions: the rules they follow (including what configuration of neighbors they look at)*
- *Environment: the environmental specifications you used (grid size, etc.)*



***Link to my model: http://tinyurl.com/haolin-eco-model-emoji***

- Overall question

  This model is built to represent a simple ecosystem with **Plant** (grass), **Herbivores** (rabbits) and **Carnivores** (foxes)

- Agents and Rules

  - **Plants**
    - A 3% of chance to reproduce themselves (duplicate to a nearby blank cell)
  - **Herbivores**
    - A 10% chance to move to a nearby plant cell in search of food, leaving a blank cell behind
    - A 20% chance to move to a nearby plant cell, leaving a grass cell behind
    - A 10% chance to move from a blank cell to a nearby blank cell
    - A 7% chance to reproduce (duplicate to a nearby blank cell)
    - A 4% chance to die (turn into a blank cell)
  - **Carnivores**
    - A 30% chance to transfer from a plant cell to a nearby plant cell (in chase of prey)
    - A 30% chance to move from a blank cell to a nearby blank cell
    - A 20% chance to hunt a herbivore and leave a blank cell behind
    - A 10% chance to reproduce (duplicate to a nearby blank cell)
    - A 4% chance to die (turn into a blank cell)

- Environment:

  - Grid size: 50 times 50

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126221817738.png" alt="image-20240126221817738" style="zoom: 25%;" />

## Part 2: Netlogo

### Problem 2A

- To Make the ants consume from multiple piles simultaneously:

  *set the evaporation rate to maximum*

  <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126194744022.png" alt="image-20240126194744022" style="zoom:50%;" />

- To make the ant colony really inefficient at consuming food:

  There are two ways of achieving this:

  to make the evaporation rate to maximum

  <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126201508094.png" alt="image-20240126201508094" style="zoom:50%;" />

  to make the diffusion rate to maximum and the evaporation rate to minimum

  <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126201532992.png" alt="image-20240126201532992" style="zoom:50%;" />

- To make the ant colony really efficient at consuming food:

  set the evaporation rate to low and diffusion rate to medium

  <img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126201624298.png" alt="image-20240126201624298" style="zoom:50%;" />

### Problem 2B

#### Adjusting the code

**before**

```Netlogo

to setup-food  ;; patch procedure
  ;; setup food source one on the right
  if (distancexy (0.6 * max-pxcor) 0) < 5
  [ set food-source-number 1 ]
  ;; setup food source two on the lower-left
  if (distancexy (-0.6 * max-pxcor) (-0.6 * max-pycor)) < 5
  [ set food-source-number 2 ]
  ;; setup food source three on the upper-left
  if (distancexy (-0.8 * max-pxcor) (0.8 * max-pycor)) < 5
  [ set food-source-number 3 ]
  ;; set "food" at sources to either 1 or 2, randomly
  if food-source-number > 0
  [ set food one-of [1 2] ]
end
```

**after**

```Netlogo
to setup-food  ;; patch procedure
  ;; setup food source one on the right
  if (distancexy (0.8 * max-pxcor) 0) < 8
  [ set food-source-number 1 ]
  ;; setup food source two on the lower-left
  if (distancexy (-0.8 * max-pxcor) (-0.8 * max-pycor)) < 3
  [ set food-source-number 2 ]
  ;; setup food source three on the upper-left
  if (distancexy (-0.8 * max-pxcor) (0.8 * max-pycor)) < 5
  [ set food-source-number 3 ]
  ;; setup food source in the middle
  if (distancexy (-0.4 * max-pxcor) (0.4 * max-pycor)) < 5
  [ set food-source-number 4 ]
  ;; set "food" at sources to either 1 or 2, randomly
  if food-source-number > 0
  [ set food one-of [1 2] ]
end

...
to recolor-patch  ;; patch procedure
  ...
    [ ...
      if food-source-number = 4 [ set pcolor yellow]
    ]
    ...
end
```

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126203454493.png" alt="image-20240126203454493" style="zoom:50%;" />

#### Observations

- For the food pile that is in the middle between another food pile and the nest, if an ant carries food from the further food pile, it would pass through the nearer food pile. When it returns from the nest, it would go to the nearer pile instead.
- Larger and nearer food piles would attract more ants and would be consumed faster than further and smaller food piles.

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126204148303.png" alt="image-20240126204148303" style="zoom:50%;" />

### Problem 2C

[Optional, skipping for now. Updates may be available afterwards]



## Part 3: ABM with Python

### Part 3A - 3E

**Complete Code**

```python
from math import *  # useful math functions
import numpy as np  # useful array objects
# (also a core scientific computing library)
import matplotlib.pyplot as plt  # nice plotting commands
from random import random, randint, choice  # random number generation functions


# Part A, midpoint function
def midpoint(t1, t2):
    return (t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2


# Part B, setup
corners = [(0, 0), (1, 0), (0.5, sqrt(3) / 2)]
N = 1000
x = np.zeros(N)
y = np.zeros(N)

# Part C, starting position
x[0] = random()
y[0] = random()

# Part D, loops
for i in range(N):
    if i == 0:
        continue
    # choose a random point from the corners list
    corner = choice(corners)
    # Calculate the midpoint and
    # save to the point list
    x[i], y[i] = midpoint((x[i - 1], y[i - 1]), corner)


# Part E, plotting the result
plt.figure()
plt.scatter(x, y)
plt.scatter((0, 1, 0.5), (0, 0, sqrt(3) / 2), color='red')  # plot the triangle vertices
plt.show()

```

### Optional

#### Adjusting # of Vertices

**4 Corners**

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126212940071.png" alt="image-20240126212940071" style="zoom: 50%;" />

**5 corners**

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126213854067.png" alt="image-20240126213854067" style="zoom:50%;" />

#### Change the fraction of the distance

**Fraction of 1/3**

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126214135313.png" alt="image-20240126214135313" style="zoom:50%;" />

#### Changing the Rule

**Not choosing the same vertex twice**

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126215413747.png" alt="image-20240126215413747" style="zoom:50%;" />



#### Adjusted Code

```python
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
```



## Part 4: Project Ideas

*What are you considering for your final project? Give an idea or two (definitely not final, just want you to start thinking about possibilities!)*

I have no firm ideas for the final project right now. But I have some former experiences with Netlogo where I created a model to represent the population flow in the two Americas:

<img src="https://rsdonkeyrepo1.oss-cn-hangzhou.aliyuncs.com/img/image-20240126220059196.png" alt="image-20240126220059196" style="zoom: 33%;" />

I would like to continue to explore modelling of population flow in specific environments like urban, coastal, etc with ABM, Cellular Automata, etc. Here are some references I found:

- [Cities and Complexity: Understanding Cities with Cellular Automata, Agent‐Based Models, and Fractals, by Michael Batty](https://www.researchgate.net/profile/Emily-Skop/publication/264488980_Inside_the_Mosaic_edited_by_Eric_Fong/links/59e4de850f7e9b0e1aa87b45/Inside-the-Mosaic-edited-by-Eric-Fong.pdf)
- [Integrated analysis of risks of coastal flooding and cliff erosion under scenarios of long term change](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s10584-008-9532-8&casa_token=rQSxlaGFNFoAAAAA:VvHk0CIR7zOlqArwj9u3IxYKBupnr5L19Z3ChaJ7rsjr33yTZsJbEa_wdKbafb_ZwmhKNia1zSJQCAy31w)
- [Multi-agent systems for the simulation of land-use and land-cover change: A review](https://www.tandfonline.com/doi/abs/10.1111/1467-8306.9302004?casa_token=hkuFCmfHwAEAAAAA:3eQcziDVT31RCeU7y6AaBu-HPUjdtORuQphT0o5-Ru23iR-MyNTvLhkqZWni0Srlxue0Qu_TeItxUQ)
- [High-resolution integrated modeling of the spatial dynamics of urban and regional systems](https://www.sciencedirect.com/science/article/pii/S0198971500000120?casa_token=7JLNr8Vn_44AAAAA:0ZU-xoHnh7S-lioPOy-D_xLzCs0QftD4_oiwaY9KtFp1InQfv_2P85ZJqc0KMuBx8k4XzRbsiQ)