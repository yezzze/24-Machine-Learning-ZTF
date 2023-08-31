import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__ (self, w1, w2, b1, rate):
        self.w1 = np.array(w1)
        self.w2 = np.array(w2)
        self.b1 = b1
        self.rate = rate 

    def Sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    
    def forward(self, i0):
        i0 = np.array(i0, ndmin=2).T
        h1 = np.dot(self.w1, i0) + self.b1[0]
        h1 = self.Sigmoid(h1) 
        o1 = np.dot(self.w2, h1) + self.b1[1]
        o1 = self.Sigmoid(o1)
        return [o1, h1]

    def backward(self, i0, o0):
        target = np.array(o0, ndmin = 2).T
        o1, h1 = self.forward(i0)[0], self.forward(i0)[1]
        delta = target - o1
        self.w2 += self.rate * np.dot((delta * o1 * (1.0 - o1)), np.array(h1, ndmin=2).T)
        delta2 = np.dot(self.w2.T, delta * o1 * (1.0 - o1))
        self.w1 += self.rate * np.dot((delta2 * h1 * (1.0 - h1)), np.array(i0, ndmin=2))

if __name__ == "__main__":
    rate = 0.5
    w1, w2, b1 = [[0.15, 0.20],[0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]], [0.35, 0.6]
    model = MLP(w1, w2, b1, rate)
    for i in range(1000):
        model.backward([0.05, 0.1], [0.01, 0.99])
        #print(mod.w1)
    print("rate = ",rate)
    print(model.forward([0.05, 0.1])[0])