import torch.nn as nn

#定义网络结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # AlexCONV1(3,96, k=11,s=4,p=0)
            nn.MaxPool2d(kernel_size=2, stride=2),# AlexPool1(k=3, s=2)
            nn.ReLU(inplace=True)
         )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #AlexCONV2(96, 256,k=5,s=1,p=2)
            nn.MaxPool2d(kernel_size=2,stride=2),  #AlexPool2(k=3,s=2)
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()
        self.out = nn.Sequential(
            nn.Linear(256*3*3, 1024),  #AlexFC6(256*6*6, 4096)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),  # AlexFC6(4096,4096)
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)  # AlexFC6(4096,1000)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
        x = self.out(x)
        return x