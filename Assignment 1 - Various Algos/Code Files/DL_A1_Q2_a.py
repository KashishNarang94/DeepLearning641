
# coding: utf-8

# In[389]:


import pandas as pd
data = pd.read_csv("q2a_dataset.csv")
data


# In[390]:


label = list(data["Label"])
data = data.drop(["Label"], axis = 1)
data


# In[391]:


w = []
w.append([[0.5]*2]*8)
w.append([[1]*4]*2)
w.append([1, 1])
b = []
b.append([0.5]*8)
b.append([-3]*2)
b.append(1)


# In[392]:


print(w)


# In[393]:


print(b)


# In[394]:


from random import uniform
for i in range(0, len(w[0])):
    for j in range(0,2):
        num = uniform(-1,1)
        num = round(num,2)
        w[0][i][j] = num

for i in range(0, len(b[0])):
    num = uniform(-1,1)
    num = round(num,2)
    b[0][i] = num


# In[395]:


print(w)


# In[396]:


print(b)


# In[397]:


import numpy as np
def compute_hidden_layer1(w, b, index):
    z = [0]*8
    o = [-1]*8
    for i in range(0, 8):
        we = np.array(w[0][i])
        x = np.array(list(data.iloc[index]))
        t = np.dot(we.T, x)
        t = t + b[0][i]
        z[i] = t
        if(t>=0):
            o[i] = 1
        else:
            o[i] = -1
    return z, o


# In[398]:


def compute_hidden_layer2(w, b, o1):
    z = [0]*2
    o = [-1]*2
    for i in range(0, 2):
        we = np.array(w[1][i])
        x = np.array(o1[(i*4):(i*4+4)])
        t = np.dot(we.T, x)
        t = t + b[1][i]
        z[i] = t
        if(t>=0):
            o[i] = 1
        else:
            o[i] = -1
    return z, o


# In[399]:


def compute_output(w, b, o):
    we = np.array(w[2])
    x = np.array(o)
    t = np.dot(we.T, x)
    t = t + b[2]
    if(t>=0):
        return [t,1]
    else:
        return [t,-1]


# In[400]:


for i in range(0, 1):
    z1, o1 = compute_hidden_layer1(w,b,i)
    print(o1)
    z2, o2 = compute_hidden_layer2(w,b,o1)
    print(o2)
    z3, o3 = compute_output(w,b,o2)
    print(o3)


# In[401]:


# parameter = 0.01
# for epoch in range(20):
#     print(epoch)
#     for i in range(0, len(data)):
# #         print("==============",i,"=============")
#         z1, o1 = compute_hidden_layer1(w,b,i)
#         z2, o2 = compute_hidden_layer2(w,b,o1)
#         z3, o3 = compute_output(w,b,o2)
#         temp = [0]*len(z1)
#         for j in range(0, len(z1)):
#             temp[j] = abs(z1[j])
#         if(o3!=label[i] and label[i]==-1):
# #             print("First")
#             for j in range(0, 8): # 4 is number of Neurons
#                 if(z1[j]>0):
#                     w[0][j] = list(np.add(w[0][j], (((parameter) * (-1-z1[j])) * np.array(list(data.iloc[i])))))
#                     b[0][j] = b[0][j] + (parameter) * (-1-z1[j])
#         elif(o3!=label[i] and label[i]==1):
# #             print("Second")
#             minimum = min(temp)
#             index = -1
#             for j in range(0, len(temp)):
#                 if(temp[j]==minimum):
#                     index = j
#                     break
# #             print(z1)
# #             print(index)
#             w[0][index] = list(np.add(w[0][index], (((parameter) * (1-z1[index])) * np.array(list(data.iloc[i])))))
#             b[0][index] = b[0][index] + (parameter) * (1-z1[index])
#         else:
#             continue
#     pcount = 0
#     ncount = 0
#     for i in range(0, len(data)):
#         _, t = compute_hidden_layer1(w,b,i)
#         _, t1 = compute_hidden_layer2(w,b,t)
#         _, out = compute_output(w,b,t1)
#         if(out==1):
#             pcount += 1
#         else:
#             ncount += 1
#     print(pcount, ncount)
# #         print(w)
# #         print(b)


# In[402]:


from sklearn.metrics import accuracy_score
from tqdm import tqdm
parameter = 0.001
for epoch in range(10):
    for i in range(0,len(data)):
#         print("=============", i, "================")
        z1, o1 = compute_hidden_layer1(w,b,i)
        z2, o2 = compute_hidden_layer2(w,b,o1)
        z3, o3 = compute_output(w,b,o2)
        temp = [0]*len(z1)
        for j in range(len(z1)):
            temp[j] = z1[j]
        for k in range(0, 8): # 8 is number of neurns to be updated
            if(o3 != label[i]):
                minimum = min(temp)
                index = -1
                for j in range(len(temp)):
                    if(temp[j] == minimum):
                        index = j
                        temp[index] = 1000
                        break
                t_o1 = [0]*len(o1)
                for j in range(len(o1)):
                    if(j!=index):
                        t_o1[j] = o1[j]
                    else:
                        if(o1[j]==1):
                            t_o1[j] = -1
                        else:
                            t_o1[j] = 1

                _, t_o2 = compute_hidden_layer2(w,b,t_o1)
                _, t_o3 = compute_output(w,b,t_o2)
                if(t_o3 == label[i]):
                    # Perform Adaline on INDEXth neuron
#                     print("Adaline")
                    w[0][index] = list(np.add(w[0][index], (((parameter) * (t_o1[index]-z1[index])) * np.array(list(data.iloc[i])))))
                    b[0][index] = b[0][index] + (parameter) * (t_o1[index]-z1[index])
                    break
            else:
                break
    y_pred = []
    for i in range(0, len(data)):
        _, t = compute_hidden_layer1(w,b,i)
        _, t1 = compute_hidden_layer2(w,b,t)
        _, out = compute_output(w,b,t1)
        y_pred.append(out)
    print("############# Epoch",epoch, "accuracy score",accuracy_score(y_pred, label),"#############")


# In[406]:


# data["label"] = label
# data


# In[405]:


# import seaborn as sns
# import matplotlib.pyplot as plt
# plot_x = np.array([0, 6])
# plot_y = []
# # getting corresponding y co-ordinates of the decision boundary
# for i in range(0, 8):
#     plot_y.append((-1/w[0][i][1]) * (w[0][i][0] * plot_x + b[0][i]))
# # plot_y = (-1/w[0][1]) * (w[0][0] * plot_x + b[0])
# # plot_z = (-1/w[1][1]) * (w[1][0] * plot_x + b[1])
# sns.FacetGrid(data, hue="label", size = 6).map(plt.scatter, "1", "2").add_legend()
# # plt.plot(plot_x, plot_y)
# # plt.plot(plot_x, plot_z)
# for i in range(0, 8):
#     plt.plot(plot_x, plot_y[i])
# plt.show()

