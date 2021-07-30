import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/mnist2') #the argument is the directory where we save the log files


#1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #It will run on the GPU if it is supported 
#hyper parameters
input_size = 784 #since each image is of size 28 x 28 and we will flatten this array into a 1D array
hidden_size = 100
num_classes = 10 #digits from 0 to 9
num_epochs = 2 
batch_size = 100
learning_rate = 0.01

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
example_data, example_targets = examples.next()

img_grid = torchvision.utils.make_grid(example_data) #example_data is the image passed to the grid
writer.add_image('mnist_images', img_grid) #mnist images is a label at the beginning
#writer.close() #this makes sure that all the outputs are being flushed
#sys.exit() #early exit


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
writer.add_graph(model, example_data.reshape(-1, 28*28))
writer.close()
#sys.exit()

#training loop
running_loss = 0.0
running_correct = 0.0

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
		
		running_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		running_correct += (predicted == labels).sum().item()

		#Every 100 step
		if (i+1) % 100 == 0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
			writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i) 
			writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
			running_loss = 0.0
			running_correct = 0.0
writer.close()

#testing
labels = []
preds = []
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	#loop over all the batches
	for images, labels1 in test_loader:
		images = images.reshape(-1, 28*28).to(device)
		labels1 = labels1.to(device)
		outputs = model(images)
		#torch.max will return the value and index
		_, predictions = torch.max(outputs, 1)
		n_samples += labels1.shape[0]
		n_correct += (predictions == labels1).sum().item() #for each correct prediction we add 1 to n_correct
		class_predictions = [F.softmax(output, dim=0) for output in outputs]
		preds.append(class_predictions)
		labels.append(predictions)

	preds = torch.cat([torch.stack(batch) for batch in preds]) #2D tensor
	labels = torch.cat(labels) #concatenate to a 1D tensor

	acc = 100.0 * n_correct/n_samples
	print(f'accuracy = {acc}')

	classes = range(10)
	for i in classes:
		labels_i = labels == i
		preds_i = preds[:, i] #get all samples for class i only
		writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0) #use class label as string
		writer.close()

