import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. Prepare dataset
xy = np.loadtxt(
    fname='01_logistic_regression/01_logistic_multilayer/diabetes.csv.gz', 
    delimiter=',', dtype=np.float32
)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

# 【关键改进 1】特征标准化（Standardization），让各特征量纲统一
x_data = (x_data - x_data.mean(dim=0)) / x_data.std(dim=0)

# 2. Design model
class MultiLogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(MultiLogisticRegressionModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 16)  # 可以稍微加大神经元数量
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 1)
        
        # 【关键改进 2】隐藏层使用 ReLU，避免梯度消失
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()  # 最后一层输出概率使用 Sigmoid

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = MultiLogisticRegressionModel()

# 3. Construct loss and optimizer
criterion = torch.nn.BCELoss()
# 【关键改进 3】换用 Adam 优化器，学习率设为 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

# 4. Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Evaluation, no need to calculate gradient so use torch.no_grad()!!!
with torch.no_grad():
    y_pred = model(x_data)
    predicted = (y_pred >= 0.5).float()
    accuracy = (predicted == y_data).sum().item() / y_data.size(0)
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

# 6. 可视化代码
plt.figure(figsize=(14, 5))

# 子图 1：训练损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(epoch_list, loss_list, color='b', label='BCE Loss')
plt.title('Training Loss Curve (with ReLU & Normalization)')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(True)
plt.legend()

# 子图 2：预测概率分布图
plt.subplot(1, 2, 2)
y_pred_np = y_pred.detach().numpy().flatten()
y_true_np = y_data.numpy().flatten()

plt.hist(y_pred_np[y_true_np == 0], bins=25, alpha=0.6, label='True Label: 0 (Negative)', color='skyblue')
plt.hist(y_pred_np[y_true_np == 1], bins=25, alpha=0.6, label='True Label: 1 (Positive)', color='salmon')
plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Predicted Probability Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Sample Count')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()