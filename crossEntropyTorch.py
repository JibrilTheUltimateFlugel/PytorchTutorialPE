import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0]) #here we only put the correct class label which in this case is class 0

#the size is n_samples x n_classes = 1 sample x 3 possible classes

Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) #we use raw logits without softmax here
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())

#Output
_, predictions1 = torch.max(Y_pred_good, 1) #1 means along the first dimension
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1) #tensor([0])
print(predictions2) #tensor([1])
