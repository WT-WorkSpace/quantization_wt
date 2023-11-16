import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 设置随机种子，以保证结果的可复现性
torch.manual_seed(0)

# 定义超参数
batch_size = 128
lr = 0.001
epochs = 50

# 数据预处理：将图像数据进行归一化处理
transform_train = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./量化/resnet18-QAT/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./量化/resnet18-QAT/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 定义ResNet-18模型
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 将原始的全连接层替换为适应CIFAR-10分类任务的新全连接层

# 定义损失函数和优化算法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 在GPU上训练模型（如果有可用GPU）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[epoch:%d, iter:%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0# running_loss 是这个批次内的累加损失，除以100为平均损失
torch.save(model.state_dict(), './量化/resnet18-QAT/model.pth')
print("模型已保存")

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images: %.2f %%' % (100 * correct / total))
