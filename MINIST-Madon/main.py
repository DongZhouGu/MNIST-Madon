import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import transforms
from torchnet import meter
from data.dataset import Dataset
from utils import Visualizer
import models

vis = Visualizer(env='my_wind')#为了可视化增加的内容
train_loss_meter = meter.AverageValueMeter()#为了可视化增加的内容
test_acc_meter = meter.AverageValueMeter()#为了可视化增加的内容
def train(args, model, device, train_loader, optimizer, epoch): # 还可添加loss_func等参数
    model.train() # 必备，将模型设置为训练模式
    train_loss_meter.reset()  # 为了可视化增加的内容
    for batch_idx, (data, target) in enumerate(train_loader): # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device) # 将数据存储CPU或者GPU
        optimizer.zero_grad() # 清除所有优化的梯度
        output = model(data)  # 喂入数据并前向传播获取输出
        """
       pytorch中CrossEntropyLoss是通过两个步骤计算出来的:
              第一步是计算log softmax，第二步是计算cross entropy（或者说是negative log likehood），
              CrossEntropyLoss不需要在网络的最后一层添加softmax和log层，直接输出全连接层即可。

              而NLLLoss则需要在定义网络的时候在最后一层添加log_softmax层(softmax和log层)

       总而言之：CrossEntropyLoss() = log_softmax() + NLLLoss() 
        """
        criterion = nn.CrossEntropyLoss()
        loss = criterion (output, target) # 调用损失函数计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        train_loss_meter.add(loss.item())
        if batch_idx % args.log_interval == 0: # 根据设置的显式间隔输出训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()  # 必备，将模型设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:  # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()  # 添加损失值
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计预测正确个数

    test_loss /= len(test_loader.dataset)
    test_acc_meter.add(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # 根据输入参数和实际cuda的有无决定是否使用GPU
    torch.manual_seed(args.seed)  # 设置随机种子，保证可重复
    device = torch.device("cuda" if use_cuda else "cpu")  # 设置使用CPU or GPU

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存

    dst=Dataset()
    train_loader = torch.utils.data.DataLoader(
        dst.data_train,batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dst.data_test,batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = models.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 实例化求解器
    for epoch in range(1, args.epochs + 1):  # 循环调用train() and test() 进行epoch迭代
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        vis.plot({'train_loss': train_loss_meter.value()[0]})
        vis.plot({'test_acc': test_acc_meter.value()[0]})  # 为了可视化增加的内容

if __name__ == '__main__':
    main()
