import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

input = torch.randn(1, 5, requires_grad=True)
print(input.shape)
print(input)
target = torch.empty(1, dtype=torch.long).random_(5)
print(target.shape)
print(target)
output = loss(input, target)
print(output)

# input = torch.randn(3, 5, requires_grad=True)
# print(input.shape)
# print(input)
# target = torch.randn(3, 5).softmax(dim=1)
# print(target.shape)
# print(target)
# output = loss(input, target)
# print(output)