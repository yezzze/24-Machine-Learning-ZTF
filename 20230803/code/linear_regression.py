import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = open(r"./data/ex1_data.txt")
    x = list(map(float, data.readline().split()))
    t = [1 for i in range(50)]
    y = list(map(float, data.readline().split()))
    plt.scatter(x, y, c="r",s=10)
    x = np.array([x])
    y = np.array([y])
    A = x
    print(A.shape,A.T.shape)
    '''
    B = np.dot(A.T, A)
    B = np.linalg.inv(B)
    C = np.dot(A.T, y.T)
    w = np.dot(B, C)
    '''
    w = np.dot(np.linalg.inv(np.dot(A,A.T)),np.dot(A,y.T))
    print(w)
    x = np.arange(0, 53)
    #a, b = w[0][0], w[1][0]
    a = w[0][0]
    y = a*x
    plt.plot(x, y)
    plt.show()
    #print(w)

 