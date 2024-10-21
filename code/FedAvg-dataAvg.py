'''
与FedAvg-normal相比 FedAvg-dataAvg的区别在于:
客户端数据分配更加均匀 每个客户端的数据量更加接近
在FedAvg-normal中: 每个客户端的数据量是随机分配的
如果客户端数量较少 可能会导致最后一个客户端的数据量过大
但是在这种情况下 能够模拟数据量倾斜极端情况 更好地验证算法的鲁棒性
所以在实际应用中 可以根据实际情况选择FedAvg-normal或FedAvg-dataAvg
'''
# 引入所需的库
import torch    # 引入torch库
import torch.utils.data # 引入torch.utils.data库，用于数据集加载和预处理
import torchvision.datasets as datasets # 引入torchvision.datasets库，用于加载数据集
import torch.nn as nn   # 引入torch.nn库，用于构建神经网络
from torchvision import models  # 引入torchvision.models库，用于加载模型
from torchvision.transforms import transforms   # 引入torchvision.transforms库，用于数据预处理
from torch.utils.data import DataLoader # 引入torch.utils.data库中的DataLoader类，用于数据加载

import json # 引入json库，用于读取json文件
import random # 引入random库，用于生成随机数

# 定义client类
class Client:
    def __init__(self, conf, local_model, train_dataset, id):
        self.conf = conf    # 客户端的配置
        
        if torch.cuda.is_available():   # 判断是否有cuda设备
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")   # 选择客户端的设备

        self.local_model = local_model.to(self.device)   # 获取全局模型

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.conf['batch_size'],
            shuffle=True)   # 客户端的训练数据加载器, 从配置文件中读取batch_size
        self.param = dict() # 建立客户端返回给server的参数字典

        self.client_id = id # 客户端的id
        self.train_dataset = train_dataset  # 客户端的训练数据集

        dataset_size = len(self.train_dataset)    # 获取数据集大小
        self.data_size = dataset_size    # 客户端的数据集大小
        print(f"client {self.client_id} size of data : {dataset_size}")   # 打印数据集大小

    # 定义获取数据集大小的函数
    def get_data_size(self):
        return self.data_size    # 返回数据集大小

    # 定义客户端的训练函数
    def local_train(self):
        # 定义优化器
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum'])

        # 本地模型训练
        self.local_model.train()
        for epoch in range(self.conf['local_epochs']):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device) # 将数据和标签移动到设备上
                optimizer.zero_grad()   # 优化器梯度清零
                output = self.local_model(data) # 模型前向传播
                loss = torch.nn.functional.cross_entropy(output, target)    # 使用交叉熵损失函数
                loss.backward() # 反向传播
                optimizer.step()    # 优化器更新参数
            print(f'client {self.client_id} local epoch {epoch + 1} done')   # 打印client的训练轮数
        for name, param in self.local_model.state_dict().items():
            self.param[name] = param.clone()    # 将本地训练后的模型参数存储到param字典中
        return self.param   # 返回参数和数据集大小
    
# 定义server类
class Server:
    def __init__(self, conf, global_model, eval_datasets):
        self.conf = conf    # 服务器的配置

        # 判断是否有cuda设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")   # 选择服务器的设备

        # 配置服务器端的模型
        self.global_model = global_model.to(self.device)

        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_datasets,
            batch_size=self.conf['batch_size'],
            shuffle=True)   # 服务器端的数据加载器（此处用于评估而不是用于训练）
        self.accuracy_history = []  # 服务器端的历史准确率，用于评估以及绘图
        self.loss_history = []  # 服务器端的历史损失，用于评估以及绘图
    
    # 定义服务器端的模型聚合函数
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            # 服务器端的模型参数更新
            update_per_layer = weight_accumulator[name]
            # 进行参数数据类型兼容处理：如果数据类型不匹配，则将update_per_layer转换为data的数据类型
            if data.type() != update_per_layer.type():
                update_per_layer = update_per_layer.type(data.type())
            # 更新服务器端的模型参数
            data.add_(update_per_layer)

    # 定义评估函数
    def model_evaluate(self):
        self.global_model.eval()    # 将模型切换为评估模式
        total_loss = 0  # 初始化总损失
        correct = 0 # 初始化正确的数量
        data_size = 0   # 初始化数据集大小

        # 评估模型
        for batch in self.eval_dataloader:
            data, target = batch    # 获取数据和标签
            data, target = data.to(self.device), target.to(self.device) # 将数据和标签移动到设备上(容易忽略)
            output = self.global_model(data)    # 模型前向传播
            loss = torch.nn.functional.cross_entropy(output, target)    # 使用交叉熵损失函数
            total_loss += loss.item()   # 计算总损失
            pred = output.data.max(1, keepdim=True)[1]  # 获取预测结果
            correct += pred.eq(target.data.view_as(pred)).sum()  # 将target调整为pred的形状并计算正确的数量
            data_size += data.size()[0] # 计算数据集大小

        loss_avg = total_loss / data_size
        self.loss_history.append(loss_avg)  # 将损失添加到历史损失中
        accuracy = 100 * correct / data_size
        self.accuracy_history.append(accuracy.item()) # 将准确率添加到历史准确率中
        return accuracy, loss_avg    # 返回准确率和损失

# 定义数据拆分函数
def split_dataset_randomly(dataset, num_clients):
    """
    将数据集随机分配给多个客户端，并返回每个客户端的数据子集
    :param dataset: 原始数据集
    :param num_clients: 客户端数量
    :return: 每个客户端的数据子集列表
    """
    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)  # 随机打乱索引

    # 计算每个客户端应分配的平均数据量
    avg_size = data_size // num_clients
    client_data_sizes = [avg_size] * num_clients

    # 在平均数据量的基础上，添加一定的随机性
    for i in range(data_size % num_clients):
        client_data_sizes[i] += 1  # 分配剩余的数据

    # 确保每个客户端的数据量在一定范围内波动
    for i in range(num_clients):
        if i < num_clients - 1:
            fluctuation = random.randint(-avg_size // 2, avg_size // 2)
            client_data_sizes[i] += fluctuation
            client_data_sizes[-1] -= fluctuation

    client_datasets = []
    start_idx = 0
    for size in client_data_sizes:
        end_idx = start_idx + size
        client_indices = indices[start_idx:end_idx]
        client_subset = [dataset[i] for i in client_indices]  # 直接获取数据子集
        client_datasets.append(client_subset)
        start_idx = end_idx

    return client_datasets  # 返回每个客户端的数据子集列表

# 定义数据集加载函数
def get_dataset(dir, name, num_clients):
    if name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    elif name=='cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    # 随机化训练数据集
    client_train_datasets = split_dataset_randomly(train_dataset, num_clients)
    return client_train_datasets, eval_dataset

# 开始训练
with open("./config-json/avg_conf.json",'r') as f:
    conf = json.load(f) # 读取配置文件

# 加载预训练的 ResNet18 模型
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 修改第一层卷积层，使其接受1个通道的输入
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# 分别定义一个服务端对象和多个客户端对象，用来模拟横向联邦训练场景
train_datasets,eval_datasets = get_dataset("./data/",conf["type"],conf["no_models"])
server = Server(conf, model, eval_datasets)
clients = []
client_data_sizes = []

# 判断是否有cuda设备
if torch.cuda.is_available():
    print("-----------------")
    print("cuda is available")
    print("-----------------")
else: 
    print("-----------------")
    print("cuda is not available, use cpu")
    print("-----------------")

# 创建多个客户端
for c in range(conf["no_models"]):
    clients.append(Client(conf, model, train_datasets[c], c+1))
    data_size = clients[c].get_data_size()
    client_data_sizes.append(data_size)

# 每一轮迭代，服务端会从当前的客户端集合中随机挑选一部分参与本轮迭代训练，被选中的客户端调用本地训练接口local_train进行本地训练，
# 最后服务器调用模型聚合函数model——aggregate来更新全局模型，代码如下所示：
for e in range(conf["global_epochs"]):
    # 采样k个客户端参与本轮联邦训练
    candidates = random.sample(clients,conf['k'])
    # 初始化weight_accumulator并在GPU上（如果可用）
    weight_accumulator = {}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    for name,params in server.global_model.state_dict().items():
        # 在指定设备上创建并初始化weight_accumulator中的张量
        weight_accumulator[name] = torch.zeros_like(params).to(device)
    
    # 客户端本地训练后将参数聚合并更新全局模型
    for c in candidates:
        # 确保本地训练后的模型差异在正确设备上
        diff = c.local_train()
        for name,params in server.global_model.state_dict().items():
                data_weight = c.data_size / sum(client_data_sizes)
                diff[name] = diff[name] * data_weight
                if diff[name].type() != weight_accumulator[name].type():
                    diff[name] = diff[name].type(weight_accumulator[name].type())
                else:
                    weight_accumulator[name].add_(diff[name] * c.data_size / sum(client_data_sizes))

    # 调用模型聚合函数来更新全局模型
    server.model_aggregate(weight_accumulator)

    # 将聚合后的全局模型参数传回客户端
    for c in clients:
        for name, param in server.global_model.state_dict().items():
            # 客户端的模型首先使用server下放的模型进行参数更新
            c.local_model.state_dict()[name].copy_(param.clone())

    acc,loss = server.model_evaluate()  # 评估全局模型
    print(f'global epoch {e + 1} done, accuracy: {acc}, loss: {loss}') # 打印全局模型的准确率和损失
    print('-----------------------------------')

# 保存模型
torch.save(server.global_model.state_dict(),".ResNet18_mnist.pth")

# 绘制准确率和损失曲线
import matplotlib.pyplot as plt

loss_history = server.loss_history
accuracy_history = server.accuracy_history

# 绘制损失曲线
plt.figure()
plt.plot(loss_history)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./Loss History.jpg')

# 绘制准确率曲线
plt.figure()
plt.plot(accuracy_history)
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('./Accuracy History.jpg')
plt.show()