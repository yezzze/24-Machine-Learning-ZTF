import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from datetime import datetime
import time


class FFNN(nn.Module):

    def __init__(self, input_num, hidden_size, output_num):
        super(FFNN, self).__init__()
        #定义网络结构
        self.layers1 = nn.Linear(input_num, hidden_size) #输入层
        self.layers2 = nn.Linear(hidden_size, output_num) #输出层

    def forward(self, x):
        out = self.layers1(x)
        out = torch.relu(out)
        out = self.layers2(out)
        return out

def training(net, num_epoches, learning_rate):
    loss_list = []
    acc_list = []
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) #优化器 )
    for epoch in range(num_epoches):
        print("current epoch = {}\t ".format(epoch + 1))
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28)) #数据初始化
            labels = Variable(labels)

            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                loss_list.append(loss.item())
                tt, predicts = torch.max(outputs.data, 1)
                acc = accuracy_score(labels.cpu(), predicts.cpu())
                acc_list.append(acc)
                print("current loss = {:.5f}\t current accuracy = {:.2%}".format(loss.item(), acc))
    return loss_list, acc_list


def testing(net):
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        outputs = net(images)

        tt, predicts = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicts == labels).sum()
        labels = labels.cpu()
        predicts = predicts.cpu()

        acc = accuracy_score(labels, predicts)

    print("Accuracy = %.2f" % (100 * correct / total))


if __name__ == '__main__':
    input_num = 784  # torch.Size([1, 28, 28])，28*28=784
    hidden_size = 256  # 隐藏层神经元数量
    out_num = 10  # 0到9的输出
    batch_size = 100
    num_epoch = 50
    learning_rate = 0.01

    # prepare MNIST dataset
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

    # loading data
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = FFNN(input_num, hidden_size, out_num)
    print(model)
    loss, acc = training(net=model, num_epoches=num_epoch, learning_rate=learning_rate)
    testing(net=model)
    print(len(loss), len(acc))
    plt.plot(loss)
    plt.plot(acc)
    plt.legend(['train_loss', 'train_acc'])
    plt.show()
