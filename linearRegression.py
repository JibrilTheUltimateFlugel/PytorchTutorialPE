import torch
import torch.nn as nn
import numpy as np
#Import from sklearn which supports regression datasets
from sklearn import datasets
import matplotlib.pyplot as plt
#Data preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32)) 
#Convert the numpy array dataset into a Torch tensor and ensure the type is float32 by using .astype(np.float32)
y = torch.from_numpy(y_numpy.astype(np.float32))

y=y.view(y.shape[0], 1) #single column vector
n_samples, n_features = X.shape

#1) Define Model (Single Layer)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#2) Loss and Optimizer
criterion = nn.MSELoss() #calculate mean squared error automatically
#Pass in the model parameters to optimize as well as the learning rate
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3) Training Loop
num_epochs = 100
for epoch in range(num_epochs):
	#Forward pass and loss
	y_pred = model(X) #fit the input data into the model
	loss = criterion(y_pred, y) #requires actual labels and predicted values

	#backward pass
	loss.backward()
	#update
	optimizer.step()
	optimizer.zero_grad()
	if (epoch+1) % 10 == 0:
		print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
#plot
predicted = model(X).detach().numpy() #detach the tensor to prevent this operation from being tracked in the computation graph, to set require gradient to false
#convert to numpy
plt.plot(X_numpy, y_numpy, 'ro') #ro for red dots
plt.plot(X_numpy, predicted, 'b') #b indicates blue
plt.show()
