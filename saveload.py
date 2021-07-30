import torch
import torch.nn as nn

class Model(nn.Module):
	def __init__(self, n_input_features):
		super(Model, self).__init__()
		self.linear = nn.Linear(n_input_features, 1)

	def forward(self, x):
		y_pred = torch.sigmoid(self.linear(x))
		return y_pred

model = Model(n_input_features=6)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
checkpoint = {
	"epoch": 90,
	"model_state": model.state_dict(),
	"optim_state": optimizer.state_dict()
}
# save the checkpoint
#torch.save(checkpoint, "checkpoint.pth")

# load the checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"] #access the epoch key
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)
#load the state dictionaries for the model and the optimizer
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])
print(optimizer.state_dict())

#for param in model.parameters():
#	print(param)

#FILE = 'model.pth'
#torch.save(model.state_dict(), FILE)

#model = torch.load(FILE)
#model.eval()

#loaded_model = Model(n_input_features=6)
#loaded_model.load_state_dict(torch.load(FILE))
#loaded_model.eval()

#for param in loaded_model.parameters():
#	print(param)
