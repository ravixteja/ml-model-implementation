# run this python file to visualize the movement of parameters
# towards minimum

# import the libraries
import numpy as np
from matplotlib import pyplot as plt

# define a function to minimize
def z_func(x,y):
    return np.sin(5*x) * np.cos(5*y) / 5
# we minimize the function by optimizing for x and y

# obtain the gradient
def gradient_z(x,y):
    deriv_x = np.cos(5*x) * np.cos(5*y)
    deriv_y = -1 * np.sin(5*x) * np.sin(5*y)
    return deriv_x, deriv_y

# define range of x and y values for plotting
x = np.arange(-1,1,0.01)
y = np.arange(-1,1,0.01)

# create meshgrid
X,Y = np.meshgrid(x,y)

# calculate Z values
Z = z_func(X,Y)

# initialize a starting point
current_pos = [0, -0.5, z_func(0, -0.5)]

# run the gradient descent algorithm

# set learning rate
learn_rate = 0.04

# iterate through loop
for _ in range(1000):
    x = current_pos[0]
    y = current_pos[1]
    z = current_pos[2]
    
    step_size_x = gradient_z(x,y)[0] * learn_rate
    current_pos[0] = current_pos[0] - step_size_x

    step_size_y = gradient_z(x,y)[1] * learn_rate
    current_pos[1] = current_pos[1] - step_size_y
    ax = plt.subplot(projection='3d',computed_zorder=False)
    ax.plot_surface(X,Y,Z,zorder=0,cmap='coolwarm')
    ax.scatter3D(current_pos[0],current_pos[1],
            z_func(current_pos[0],current_pos[1]),
            color='r',zorder=1)
    plt.pause(0.01)
    ax.clear()