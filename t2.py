# 使用 loss function
import torch
import torch.nn as nn


loss_fn = torch.nn.CrossEntropyLoss()

# 生成随机的 tensor
# a1 = torch.randn(3, 5)
# a2 = torch.randn(3, 5)
a1 = torch.randn(3, 5, requires_grad=True)
a2 = torch.randn(3, 5, requires_grad=True)

print(a1.grad)
print(a2.grad)

# 计算 loss
loss = loss_fn(a1, a2)
print(loss)
print(type(loss))
print(loss.grad_fn)
print(loss.grad)

loss.backward()
print(a1.grad)
print(a2.grad)
print(loss.grad_fn)
