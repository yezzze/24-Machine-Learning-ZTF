import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def up_data(data):
    x, y = [], []
    for elem in data:
        if elem[-1] == 1 : 
            x.append(elem)
            y.append(0)
        elif elem[-1] == 3:
            elem[-1]=1
            x.append(elem)
            y.append(1) 
    return x, y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(x,y):
    return y* np.log(x)+(1-y)*np.log(1-x)

def get_L(w,X,Y):
    L = 1
    l = 0
    for i,x in enumerate(X):
        h_x = sigmoid(np.dot(x,w.T))
        L *= (h_x ** Y[i])*((1-h_x)**(1-Y[i]))
        #l += f(h_x,Y[i])
        #if Y[i] == 1:
        #    L *= h_x 
        #else:
        #    L *= 1-h_x
        #print(h_x, L)
    return L

def get_w(w,X,Y):
    w_ = 0
    for i,x in enumerate(X):
        h_x = sigmoid(np.dot(x,w.T))
        w_ += (h_x-Y[i])*x
    w_ /= len(X)
    return w_

def trainning(X,Y):
    w = [random.random()*0.001 for i in range(161)]
    w = np.array(w)
    L = get_L(w,X,Y)
    delta_L = np.abs(L)
    print(L, delta_L)
    rate, round = 1e-3, 1
    f_x, f_y = [], []
    while delta_L > 0 and round <= 5000:
        if round % 100 == 0 : 
            print("# Round{round}: loss = {loss}".format(round = round, loss = float(delta_L)))
            f_x.append(round)
            f_y.append(delta_L)            
        w_ = get_w(w,X,Y)
        w -= w_*rate
        L_ = get_L(w,X,Y)
        delta_L = np.abs(L - L_)
        L = L_ 
        round += 1
    #print(w)
    cnt = 0
    acc = len(X)
    for i, x in enumerate(X):
        h_x = sigmoid(np.dot(x,w.T))
        if h_x < 0.5 and Y[i]==0: cnt += 1
        elif h_x > 0.5 and Y[i]==1: cnt += 1
        print("data{x}: H_x{x} = {tmp} Y= {y}".format(x = i+1, tmp = h_x, y = Y[i]))
    acc = cnt/acc*100
    print("Accuracy = {x}%".format(x = acc))

    plt.scatter(f_x,f_y,c="r",s=10)
    plt.plot(f_x,f_y,'orange')
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv(r"./data/ex2_data.csv")
    data = np.array(data)
    X, Y = up_data(data)
    print(X)
    trainning(X,Y)
        