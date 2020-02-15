import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),  # b, 16(高度), 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 32, 12, 12
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 128, 4, 4
        )

        self.out = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)  # (input, output)
        )

    def forward(self, x):
        x = self.layer1(x)  # (batch, 16, 26, 26) -> (batchsize, 输出图片高度, 输出图片长度, 输出图片宽度)
        x = self.layer2(x)  # (batch, 32, 12, 12)
        x = self.layer3(x)  # (batch, 64, 10, 10)
        x = self.layer4(x)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1)  # 扩展、展平 -> (batch, 128 * 4 * 4)
        x = self.out(x)
        return x

