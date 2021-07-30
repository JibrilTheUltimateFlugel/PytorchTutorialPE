import torch
import torch.nn as nn

class NeuralNet2(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet2, self).__init__()
		#archtecture
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(hidden_size, num_classes) #the output size is equal to the number of classes
	def forward(self, x):
		out = self.linear1(x)
		out = self.ReLU(out)
		out = self.linear2(out)
		#no softmax
		return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()
