import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# LeNet-5能达到98%的准确率

print(torch.__version__)
Batch_Size = 256
EPOCH = 20
Learning_rate = 0.01
device = torch.device('cuda:0')
print(device)

# 1.加载数据
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=Batch_Size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=Batch_Size, shuffle=True)

# 2.定义网络
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):   # [batch, 1, 28, 28]
        insize = x.size(0)  # insize = batch
        x = self.conv1(x)  # [batch, 6, 24, 24]
        x = F.relu(F.avg_pool2d(x, 2, 2))   # [batch, 6, 12, 12]
        x = self.conv2(x)       # [batch, 16, 8, 8]
        x = F.relu(F.avg_pool2d(x, 2, 2))   # [batch, 16, 4, 4]

        x = x.view(insize, -1)    # [batch, 256]
        x = F.relu(self.fc1(x))   # [batch, 120]
        x = F.relu(self.fc2(x))   # [batch, 84]
        x = F.relu(self.fc3(x))   # [batch, 10]
        return x

# 3.定义网络和优化器
model = LeNet5().to(device)
optimizer = optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)
criten = nn.CrossEntropyLoss().to(device)

# 4.训练模型
loss_set = []
for epoch in range(EPOCH):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        optimizer.zero_grad()
        loss = criten(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    # loss_set.append(loss.item())
    # print(loss_set)
    # 每个EPOCH测试准确率
    correct = 0
    test_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)  # data: [batch, 1, 32, 32] target: [batch, 10]
        output = model(data)
        loss = criten(output, target)
        test_loss += loss.item()

        pred = output.argmax(dim=1)
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('test loss:{:.4f}, acc:{}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), 100 * correct // len(test_loader.dataset)
    ))
