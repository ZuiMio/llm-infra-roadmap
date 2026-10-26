
import torch

## data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()                               

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)                         # 自动通过 __call__ 执行 forward
    loss = criterion(input=y_pred, target=y_data)  # loss
    print(epoch, loss)

    optimizer.zero_grad()                           # set grad 0
    loss.backward()                                 # backward
    optimizer.step()                                # update

## Output
print('w = ', model.linear.weight.item())           # use .item() to show the content in the matrix easily
print('b = ', model.linear.bias.item())

## Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
# print("y_pred =", y_test.detach())
