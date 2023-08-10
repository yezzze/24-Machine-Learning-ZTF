import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets,transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import torch.nn.functional as F

# 定义超参数
batch_size = 72
lr = 0.001
epochs = 30

# 标准化
data_tf = transforms.Compose([transforms.Resize((224,224),interpolation = InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# 读取数据
train_data = datasets.CIFAR10(root = './data',train = True,transform = data_tf)
test_data = datasets.CIFAR10(root = './data',train = False,transform = data_tf)


# 装载数据
train_loader = DataLoader(train_data,batch_size = batch_size,shuffle = True)   # shuffle打乱数据集
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle = True)

# 构建网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3,96,kernel_size = 11,stride = 4,padding = 0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)


        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256 * 5 * 5, 4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096, 1000)

    # 前向传播
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 5 * 5)  # 平铺
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x


# 参数初始化
def initial(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)     # 使得输入和输出的方差相同

# 使用cpu或gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())

net = AlexNet().to(device)
net.apply(initial)


criterion = nn.CrossEntropyLoss()  # 损失函数, 交叉熵
optimizer = optim.SGD(net.parameters(),lr = lr,momentum = 0.9,weight_decay = 5e-4)  # 优化器，使用SGD

# 训练和测试
train_loss,test_loss,train_acc,test_acc = [],[],[],[]
if __name__ == '__main__':
    save_path = './Alexnet.pth'
    for i in range(epochs):
        # 训练
        net.train()
        temp_loss, temp_correct = 0, 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()
            # print(loss)
            # 计算每次loss与预测正确的个数
            label_hat = torch.argmax(y_hat, dim=1)
            temp_correct += (label_hat == y).sum()
            temp_loss += loss

        print(f'epoch:{i + 1}  train loss:{temp_loss / len(train_loader):.3f}, train Aacc:{temp_correct / 50000 * 100:.2f}%',
              end='\n')
        torch.save(net.state_dict(), save_path)
        train_loss.append((temp_loss / len(train_loader)).item())
        train_acc.append((temp_correct/50000).item())

    # 测试
    temp_loss, temp_correct = 0, 0
    net.to(device)
    net.eval()
    net.load_state_dict(torch.load('./Alexnet.pth'))
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)

            label_hat = torch.argmax(y_hat, dim=1)
            temp_correct += (label_hat == y).sum()
            temp_loss += loss

        print(f'test loss:{temp_loss / len(test_loader):.3f}, test acc:{temp_correct / 100:.2f}%')
        test_loss.append((temp_loss / len(test_loader)).item())
        test_acc.append((temp_correct / 10000).item())
    # acc = 80.14

