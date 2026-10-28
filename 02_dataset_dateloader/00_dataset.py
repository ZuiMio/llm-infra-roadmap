

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt  # Fixed: imported pyplot correctly

# Prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(fname=filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('01_logistic_regression/01_logistic_multilayer/diabetes.csv.gz')
train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# Design model
class MultiLogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(MultiLogisticRegressionModel, self).__init__()
        # 8 input dimensions, matching the diabetes dataset features
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x
model = MultiLogisticRegressionModel()

# Construct loss and optimizer
criterion = torch.nn.BCELoss(size_average=False)    # 二元交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training cycle
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1. Prepare data
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()

    x = np.linspace(0, 10, 200)
    x_t = torch.Tensor(x).view((200, 1))
    y_t = model(x_t)
    y = y_t.data.numpy()
    plt.plot(x, y)
    plt.plot([0, 10], [0.5, 0.5], c='r')
    plt.xlabel('Hours')
    plt.ylabel('Probability of Pass')
    plt.grid()
    plt.show()
###########################################################################################
# import torchvision
# train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)
# test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=True)
# train_set = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True)
# • MNIST
# • Fashion-MNIST
# • EMNIST
# • COCO
# • LSUN
# • ImageFolder
# • DatasetFolder
# • Imagenet-12
# • CIFAR
# • STL10
# • PhotoTour
###########################################################################################
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision import datasets
# train_dataset = datasets.MNIST(root=’../dataset/mnist’,
#             train=True,
#             transform= transforms.ToTensor(),
#             download=True) 
# test_dataset = datasets.MNIST(root=’../dataset/mnist’,
#         train=False,
#         transform= transforms.ToTensor(),
#         download=True)
# train_loader = DataLoader(dataset=train_dataset,
#     batch_size=32,
#     shuffle=True)
# test_loader = DataLoader(dataset=test_dataset,
#     batch_size=32,
#     shuffle=False)
# for batch_idx, (inputs, target) in enumerate(train_loader):
#     pass
###########################################################################################