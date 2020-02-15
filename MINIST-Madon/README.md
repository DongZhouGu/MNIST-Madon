├── data/
│   ├── MNIST
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── AlexNet.py
│   ├── MyNet.py
│   └── Net.py
└── utils/
│   ├── __init__.py
│   └── visualize.py
├── main.py
├── requirements.txt
├── README.md

data/：MNIST数据集存放，数据预处理等
models/：模型定义，可以有多个模型，例如上面的AlexNet，MyNet，等一个模型对应一个文件
utils/：可能用到的工具函数，在本次实验中主要是封装了可视化工具
main.py：主文件，训练和测试程序的入口，可通过argparse来指定不同的参数
requirements.txt：程序依赖的第三方库
README.md：提供程序的必要说明

