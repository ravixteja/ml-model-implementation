# run this python file to visualize the movement of parameter
# towards minimum

import numpy as np
import matplotlib.pyplot as plt

def y_func(x):
    return x ** 2

def dydx(x):
    return 2 * x

x = np.arange(-100,100,0.1)
y = y_func(x)

curr_pos = [95,y_func(95)]
n = 0.01

for _ in range(1000):
    curr_pos[0] = curr_pos[0] - (n * dydx(curr_pos[0]))
    curr_pos[1] = y_func(curr_pos[0])

    plt.plot(x,y)
    plt.scatter(curr_pos[0],curr_pos[1],color='r')
    plt.pause(0.01)
    plt.clf()