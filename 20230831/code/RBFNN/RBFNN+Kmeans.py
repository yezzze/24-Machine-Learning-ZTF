import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random

def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data, k, max_iter=1000):
    centers = random.sample(list(data),k)  # 初始聚类中心
    # 开始迭代
    predict_l=[]
    for i in range(max_iter):  # 迭代次数
        print("第{}次迭代".format(i+1))
        clusters, predict_l = {}, []    # 聚类结果
        # 初始化
        for j in range(k): clusters[j] = []
        for sample in data:  # 遍历每个样本
            dist = []  # 计算样本到每个聚类中心的距离
            for c in centers:  # 遍历聚类中心
                dist.append(distance(sample, c)) 
            id = np.argmin(dist)  # 最小距离的索引
            clusters[id].append(sample)   # 将该样本添加到第id个聚类中心
            predict_l.append(id)
        pre_centers = centers.copy()  # 记录之前的聚类中心点

        # 重新计算中心点（计算该聚类中心的所有样本的均值）
        for c in clusters.keys():
            centers[c] = np.mean(clusters[c], axis=0)
  
        flag = True
        for c in range(k):
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                flag = False
                break
        # 如果新旧聚类中心不变，则迭代停止
        if flag == True: break
    return np.array(centers)

class RBFNN(object):
    def __init__(self,input_num,mu_num,out_num,round_num,rate):
        self.input_num = input_num
        self.mu_num = mu_num
        self.out_num = out_num 
        self.round_num = round_num
        self.rate = rate
        self.mu = 0
        self.sigma = 0
        self.w = np.random.normal(0,1,(out_num,mu_num))

    def gaus(self,x,mu,sigma):
        return np.exp(-sum(np.multiply((x-mu),x-mu))/(2*(sigma**2)))

    def cal_hidden(self,x): #计算隐藏层
        hidden = np.zeros((self.mu_num,),dtype=float)
        for id,mu in enumerate(self.mu):
            hidden[id]=self.gaus(x,mu,self.sigma[id])
        return np.mat(hidden).T

    def backward(self, data, rate): #梯度下降
        delta_mu, delta_sigma, delta_w = 0, 0, 0
        loss = []
        for x in data: #遍历每一组训练数据
            hidden = self.cal_hidden(x[:-1])
            
            pre_y = np.dot(self.w,hidden)
            delta_y = x[-1]-pre_y
            loss.append(delta_y**2)
            #print(delta_y**2)
            
            delta_w_ba = np.multiply(delta_y,hidden)

            delta_x_mu = np.mat([x[:-1]-mu for mu in self.mu])
            #print(delta_x_mu)
            delta_mu_ba = np.multiply(delta_w_ba,delta_x_mu)
            #delta_mu_ = np.multiply(hidden,delta_x_mu)
            #print(delta_y,"\n",hidden,"\n",delta_x_mu)
            #print(delta_mu_ba)

            delta_x_mu_square = np.mat([[sum((x[:-1]-mu)**2)] for mu in self.mu])
            delta_sigma_ba = np.multiply(delta_w_ba, delta_x_mu_square)
            
            # delta求和
            delta_w += delta_w_ba
            #print(delta_w)
            delta_mu += delta_mu_ba
            delta_sigma += delta_sigma_ba
        #更新
        self.mu += np.multiply((self.w.T)/(self.sigma**2),delta_mu*rate)
        self.sigma += np.multiply((self.w.T)/(self.sigma**3),delta_sigma*rate)
        self.w += (delta_w*rate).T 
        return sum(loss)/2

    def predict(self,x):
        hidden = self.cal_hidden(x)
        y = np.dot(self.w,hidden)
        return float(y)

    def metrics(self, a, b):
        return np.linalg.norm(a - b, ord = 2)

    def elem_init(self,data): #中间层和sigma的数据初始化
        train_ba = data[:,:-1] 
        self.mu = k_means(train_ba, self.mu_num) #用K_means跑出中心点
        #print(self.mu)
        max_dis = float(0)
        for i in range(0, self.mu_num):
            for j in range(i + 1, self.mu_num):
                max_dis = max(max_dis, self.metrics(self.mu[i], self.mu[j]))

        sigma_init = float(max_dis / ((2 * self.mu_num) ** 0.5))
        self.sigma = [[sigma_init] for i in range(self.mu_num)]
        self.sigma = np.array(self.sigma)
        #self.sigma = np.mat([sigma_init for i in range(self.mu_num)])
        #print((self.w.T).shape)
        #print(self.sigma)
        #print((self.sigma**2))
        #print((self.w.T/(self.sigma**2)).shape)        

    def training(self,data):
        round, loss = 1, []
        while round <= self.round_num:
            loss_now = self.backward(data, self.rate)
            #输出每一轮的loss
            print("# Round{num}: loss = {loss}".format(num = round, loss = float(loss_now)))
            loss.append(float(loss_now))
            round += 1
        return loss

    def solve(self, train, test):
        self.elem_init(train)
        centers = self.mu

        loss = self.training(train)

        clr1, clr2, error = [], [], []
        for x in test: #对test数据做预测，并检验是否正确
            y = int(self.predict(x[:-1])+0.5)
            if y == 1 and y == int(x[-1]):
                clr1.append(x[:-1])
            elif y == 2 and y == int(x[-1]):
                clr2.append(x[:-1])
            else:
                error.append(x[:-1])
        clr1, clr2, error = np.array(clr1), np.array(clr2), np.array(error)
        print("red_num =",len(clr1),"blue_num =",len(clr2),"erro_num =",len(error))
        plt.scatter(clr1[:, 0], clr1[:, 1], c = 'r', s=15)
        plt.scatter(clr2[:, 0], clr2[:, 1], c = 'b', s=15)
        if len(error): plt.scatter(error[:, 0], error[:, 1], c = 'g', s = 15)
        plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 15)
        plt.show()
        x = np.array([i for i in range(1,len(loss)+1)])
        loss = np.array(loss)
        plt.scatter(x,loss,c="r",s=2)
        plt.plot(x,loss,'b')
        plt.show()

if __name__ == '__main__':
    size = 300     #训练数据的size     
    round= 100     #梯度下降轮数     
    rate = 0.0001  #学习率 
    centers_num = 30

    data_x=open(r'moon data/X.txt')
    data_y=open(r'moon data/Y.txt')
    data = []
    for i in range(2000):
        x, y = map(float, data_x.readline().split())
        tag = int(data_y.readline())
        data.append([x,y,tag])
    data=np.array(data)
    data=shuffle(data)

    train_data, test_data = np.array(data[:size]), np.array(data[size:])
    ans = RBFNN(2,centers_num,1,round,rate)
    ans.solve(train_data, test_data)