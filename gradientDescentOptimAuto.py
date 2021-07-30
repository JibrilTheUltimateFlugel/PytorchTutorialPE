import torch
import torch.nn as nn
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32) #Create an Xtest tensor for testing
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

#Model
model = nn.Linear(input_size,output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}') #call the item() method since X_test consists of only one item to obtain the float value

#Training Loop
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss() #Mean Squared Error loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(n_iters):
	#forward pass
	y_pred = model(X)
	l = loss(Y, y_pred)
	l.backward()
	optimizer.step()
	optimizer.zero_grad() #To empty gradient after each iteration step
	if epoch % 10 == 0: #print at every 10 step
		[w,b] = model.parameters()
		print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
