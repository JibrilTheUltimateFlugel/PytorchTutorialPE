import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#dataset has PILImage images of range [0,1]
#Transform to Tensors with normalized range [-1, 1]

#Composite Transformation To Tensor then Normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) #the input channel size is 3 since we have 3 color channels
		self.pool = nn.MaxPool2d(2, 2) #kernel size of 2 indicating 2 x 2 boxes, and stride of 2 which means sliding the window by 2
		self.conv2 = nn.Conv2d(6, 16, 5) #the input of this layer must match output of previous convolutional layer
		self.fc1 = nn.Linear(16*5*5, 120) #fully connected layer
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10) #the final output class has to be 10 since we have 10 different classes

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x))) #Inline activation function and pooling layer
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		#No softmax since it is included in the loss
		return x

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images,labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		#Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		#Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 2000 == 0:
			print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
print('Finished Training')

#Evaluation
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	n_class_correct = [0 for i in range(10)]
	n_class_samples = [0 for i in range(10)]
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		
		_, predicted = torch.max(outputs, 1)
		n_samples += labels.size(0)
		n_correct += (predicted == labels).sum().item()

		for i in range(batch_size):
			label = labels[i]
			pred = predicted[i]
			if (label == pred):
				n_class_correct[label] += 1
			n_class_samples[label] += 1
	acc = 100.0 * n_correct/n_samples
	print(f'Accuracy: {acc} %')
	for i in range(10):
		acc = 100.0 * n_class_correct[i] / n_class_samples[i]
		print(f'Accuracy of {classes[i]}: {acc} %')

