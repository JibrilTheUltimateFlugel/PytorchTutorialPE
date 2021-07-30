import torch
import torch.nn as nn
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
	return w * x
#Training Loop
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss() #Mean Squared Error loss
optimizer = torch.optim.SGD([w], lr=learning_rate)
for epoch in range(n_iters):
	#forward pass
	y_pred = forward(X)
	l = loss(Y, y_pred)
	l.backward()
	optimizer.step()
	optimizer.zero_grad() #To empty gradient after each iteration step
	if epoch % 10 == 0: #print at every 10 step
		print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')
