# -*- coding: utf-8 -*-


import seaborn as sns
import matplotlib.pyplot as plt
# It is Used to plot the decision Boundary within scattered data points
def plot_data(w, b, data, title): 
    plot_x = np.array([-2, 3])
    # getting corresponding y co-ordinates of the decision boundary
    plot_y = (-1/w[1]) * (w[0] * plot_x + b)
    sns.FacetGrid(data, hue="label", size = 4).map(plt.scatter, 0, 1).add_legend()
    plt.plot(plot_x, plot_y)
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.title(title)
    plt.show()

import numpy as np
# Implementation of PTA Algorithm
def PTA(data_list, data_label, data, title):
    w = [0, 0]
    b = 0
    w = np.array(w)
    count = 1
    while(True): # Iterating till there is no Error
        flag = 0
        print("Step ",count,"-> Weights :",w,", Bias :", b)
        plot_data(w, b, data, title)
        for i in range(0, len(data_list)):
            mat1 = np.array(data_list[i])
            t = np.dot(w.T, mat1)
            t += b
            t = t * data_label[i]
            if(t <= 0): # Error Condition
                w = np.add(w, (np.array(data_list[i])*(data_label[i]))) #Updating the weights by w = w + yx
                b += (1*data_label[i])
                flag = 1
                break
        if(count==30):
            break
        count += 1
        if(flag == 0):
            break
    print("\nFinal Weights : ", w)
    print("Final Bias :", b)
    return w,b

"""### PTA for OR operation"""

import pandas as pd
or_list = [[0,0], [0,1], [1,0], [1,1]]
or_label = [-1, 1, 1, 1]

or_data = pd.DataFrame(or_list)
or_data["label"] = or_label
or_data

# Calling PTA Algortihm for OR operation
w, b = PTA(or_list, or_label, or_data, "Decision Boundary for OR operation")

# Plotting the final Decision Boundary after convergence
plot_data(w, b, or_data, "Decision Boundary for OR operation")

"""### PTA for XOR operation"""

xor_list = [[0,0], [0,1], [1,0], [1,1]]
xor_label = [-1,1,1,-1]
xor_data = pd.DataFrame(xor_list)
xor_data["label"] = xor_label
xor_data

w, b = PTA(xor_list, xor_label, xor_data, "Decision Boundary for XOR operation")

# plot_data(w, b, xor_data, "Decision Boundary for XOR operation")

"""#### XOR is not Converging Using PTA
1. By running algorithm for 15 steps we can see that 8-11, 12-15,... the weights in these steps are repeating (So we can say that these 4 weights are repeating itself).
2. Step 8 to 11 are being repeated over the iterations, hence they are not converging.

### PTA for AND operation
"""

and_list = [[0,0], [0,1], [1,0], [1,1]]
and_label = [-1,-1,-1,1]
and_data = pd.DataFrame(and_list)
and_data["label"] = and_label
and_data

# Calling PTA Algorithm on AND Data
w, b = PTA(and_list, and_label, and_data, "Decision Boundary for AND operation")

# Plotting the final Decision Boundary after convergence
plot_data(w, b, and_data, "Decision Boundary for AND operation")

"""### PTA for NOT Operation"""

not_list = [[0], [1]]
not_label = [1,-1]
not_data = pd.DataFrame(not_list)
not_data["label"] = not_label
not_data

# Plotting 1D data with decision Boundary
def plot_not_data(w, b, not_data, title):
    xmin, xmax = plt.xlim()
    a = -b / w[0]
    xx = np.linspace(xmin, xmax)
    yy = a * xx - (b) / w[0]
    plt.scatter([0,0], not_data[0])
    plt.plot(xx, yy, 'k-')
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.title(title)
    plt.show()

# PTA Algorithm
w = [0]
b = 0
w = np.array(w)
count = 1
while(True): # Iterating till there is no Error
    flag = 0
    print("Step ",count,"-> Weights :",w,", Bias :", b)
    plot_not_data(w, b, not_data, "Decision Boundary for NOT operation")
    for i in range(0, len(not_list)):
        mat1 = np.array(not_list[i])
        t = np.dot(w.T, mat1)
        t += b
        t = t * not_label[i]
        if(t <= 0): # Error Condition
            w = np.add(w, (np.array(not_list[i])*(not_label[i]))) # Updating the weights by w = w + yx
            b += (1*not_label[i])
            flag = 1
            break
    count += 1
    if(flag == 0):
        break

# Plotting the final Decision Boundary after convergence
plot_not_data(w, b, not_data, "Decision Boundary for NOT operation")