import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #It will run on the GPU if it is supported 
#hyper parameters
input_size = 784 #since each image is of size 28 x 28 and we will flatten this array into a 1D array
hidden_size = 100
num_classes = 10 #digits from 0 to 9
num_epochs = 2 
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		#Layers
		self.l1 = nn.Linear(input_size, hidden_size) #input size as input and hidden size as output
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(hidden_size, num_classes)
	#forward propagation
	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		#no softmax here since we will use the Cross-Entropy loss
		return out

model = NeuralNet(input_size, hidden_size, num_classes)
#loss and optimizer
criterion = nn.CrossEntropyLoss() #this includes softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		#The image tensor needs the size to be 100, 784 since the input size is 784
		#reshape the tensor
		images = images.reshape(-1, 28*28).to(device) #-1 to find dimension automatically
		labels = labels.to(device)
		#forward pass
		outputs = model(images)
		loss = criterion(outputs, labels) #compare between outputs and actual labels
		#back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step() #update step

		#Every 100 step
		if (i+1) % 100 == 0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#testing
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	#loop over all the batches
	for images, labels in test_loader:
		images = images.reshape(-1, 28*28).to(device)
		labels = labels.to(device)
		outputs = model(images)
		#torch.max will return the value and index
		_, predictions = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (predictions == labels).sum().item() #for each correct prediction we add 1 to n_correct
	acc = 100.0 * n_correct/n_samples
	print(f'accuracy = {acc}')
