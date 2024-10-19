# 引入所需的库
import torch    # 引入torch库
import torch.utils.data # 引入torch.utils.data库，用于数据集加载和预处理
import torchvision.datasets as datasets # 引入torchvision.datasets库，用于加载数据集
from torchvision import models  # 引入torchvision.models库，用于加载模型
from torchvision.transforms import transforms   # 引入torchvision.transforms库，用于数据预处理

import json # 引入json库，用于读取json文件
import random # 引入random库，用于生成随机数

# 定义client类
class Client:
    def __init__(self, conf, model, train_dataset, id=1):
        self.conf = conf    # 客户端的配置
        
        if torch.cuda.is_available():   # 判断是否有cuda设备
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")   # 选择客户端的设备

        self.local_model = model.to(self.device)    # 客户端的模型
        self.global_model = models.get_model(self.conf['model_name']).to(self.device)   # 获取全局模型

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.conf['batch_size'],
            shuffle=True)   # 客户端的训练数据加载器, 从配置文件中读取batch_size
        self.param = dict() # 建立客户端返回给server的参数字典

        self.train_dataset = train_dataset  # 客户端的训练数据集
        self.client_id = id # 客户端的id

    # 定义客户端的训练函数
    def local_train(self, model):
        for name, param in model.state_dict().items():
            # 客户端的模型首先使用server下放的模型进行参数更新
            self.local_model.state_dict()[name].copy_(param.clone())

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
            print(f'client {self.client_id} local epoch {epoch} done')   # 打印client的训练轮数
        for name, param in self.local_model.state_dict().items():
            self.param[name] = param.clone()    # 将本地训练后的模型参数存储到param字典中
        return self.param   # 返回参数字典
    
# 定义server类
class Server:
    def __init__(self, conf, eval_datasets):
        self.conf = conf    # 服务器的配置
        # 配置服务器端的模型
        self.global_model = models.get_model(self.conf['model_name'])
        # Server类中的global_model需要被移到device上

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")   # 选择服务器的设备

        self.global_model = models.get_model(self.conf['model_name']).to(self.device)   # 获取全局模型
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
            update_per_layer = weight_accumulator[name] * self.conf['lambda']
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
        for batch_id, batch in enumerate(self.eval_dataloader):
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
        return accuracy, loss_avg    # 返回准确率和损失
    
# 定义数据集加载函数
def get_dataset(dir, name):
    if name == 'mnist':
        train_dataset = datasets.MINST(dir, train=True, download=True,transform=transforms.ToTensor())
        eval_dataset = datasets.MINST(dir, train=False, transform=transforms.ToTensor())
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
    return train_dataset, eval_dataset

# 开始训练
with open("conf.json",'r') as f:
    conf = json.load(f) # 读取配置文件

# 分别定义一个服务端对象和多个客户端对象，用来模拟横向联邦训练场景
train_datasets,eval_datasets = get_dataset("./data/",conf["type"])
server = Server(conf,eval_datasets)
clients = []

# 创建多个客户端
for c in range(conf["no_models"]):
    clients.append(Client(conf,server.global_model,train_datasets,c))
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

    for c in candidates:
        # 确保本地训练后的模型差异在正确设备上
        diff = c.local_train(server.global_model)
        for name,params in server.global_model.state_dict().items():
            weight_accumulator[name].add_(diff[name])

    server.model_aggregate(weight_accumulator)
    acc,loss = server.model_evaluate()
    print(f'global epoch {e} done, accuracy: {acc}, loss: {loss}')