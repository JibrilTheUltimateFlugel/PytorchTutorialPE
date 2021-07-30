import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#custom dataset
class WineDataset(Dataset):
	def __init__(self):
		xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
		self.x = torch.from_numpy(xy[:, 1:]) #skips the very first column from index 1 to the end
		self.y = torch.from_numpy(xy[:, [0]]) #the dimensions for this is n_samples by 1
		self.n_samples = xy.shape[0] #the first number in xy is the number of samples

	def __getitem__(self,index):
		#this allows for indexing later
		return self.x[index], self.y[index]

	def __len__(self):
		#this allows us to obtain the length of the dataset
		return self.n_samples

#Create dataset
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

#Create a training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4) #total number of samples divided by batch size
print(total_samples, n_iterations)

for epoch in range(num_epochs): 
	#loop over the whole dataloader
	for i, (inputs, labels) in enumerate(dataloader):
		#forward pass, backward pass and update
		if (i+1) % 5 == 0: #show information every 5 step
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
