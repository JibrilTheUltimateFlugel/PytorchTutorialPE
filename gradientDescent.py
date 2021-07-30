import torch
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
	return w * x
def loss(y, y_predicted):
	return ((y_predicted-y)**2).mean()
#Training Loop
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
	#forward pass
	y_pred = forward(X)
	l = loss(Y, y_pred)
	#gradients = backward pass
	l.backward() #this will let pytorch calculate dl/dw automatically
	#update weights this should not be part of the computational graph thus we wrap it
	with torch.no_grad():
		w -= learning_rate * w.grad
	#zero the gradient again since everytime we call backward() the gradient will be accumulated in w.grad attribute
	w.grad.zero_()
	if epoch % 10 == 0: #print at every 10 step
		print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')
