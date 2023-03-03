import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x0 = 0
y0 = 0
yx0 = 0
L = 16
linspace_size_from = x0 - L
linspace_size_to = x0 + L
steps = 1000
dx = (linspace_size_to - x0) / (steps - 1)
n_iterr = 1000
learning_rate = 0.03
neurons = 100
extralayers = 0
alpha = 0.001

x_R = torch.linspace(x0, linspace_size_to, steps)[:, None]
x_L = torch.linspace(linspace_size_from, x0 - dx, steps - 1)[:, None]
x = torch.cat((x_L, x_R), 0)
xx = x
x.requires_grad = True

class Model(nn.Module):
    def __init__(self, x0, y0, yx0, input_dim, hidden_dim ,output_dim, layers):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.yx0 = yx0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layer = []
        for i in range(self.layers):
            self.hidden_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias = True))
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)  
        self.activation = nn.Sigmoid()
    
    def forward_prop(self, x):
        x = self.input_layer.forward(x)
        x = self.activation(x)
        for i in range(0, self.layers):
            x = self.hidden_layer[i].forward(x)
            x = self.activation(x)
        x = self.output_layer.forward(x)
        return x
    
    def loss(self, x, y_pred):
        y_pred_x = torch.autograd.grad(y_pred, x, torch.ones_like(x), create_graph = True)[0]
        #self.yx = self.y_pred_x.detach()
        #y_pred_xx = torch.autograd.grad(y_pred_x, x, torch.ones_like(x), retain_graph=True)[0]
        #self.yxx = self.y_pred_xx.detach()
        self.L_diff = torch.mean((y_pred_x - torch.cos(x))**2)
        self.L_cond = (y_pred[[steps - 1]] - self.y0)**2
        loss = self.L_diff + alpha*self.L_cond
        return loss
    
    
model = Model(x0, y0, yx0, 1, neurons , 1, extralayers)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for epoch in range(n_iterr):
    y_pred = model.forward_prop(x)
    loss = model.loss(x, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Loss: ", loss.item())
    
with torch.no_grad():
    print("initial condition: ", y_pred[[steps - 1]])
    print("L_diff: ", model.L_diff.item())
    print("L_cond: ", model.L_cond.item())
    plt.plot(xx.numpy(), y_pred.numpy())
    plt.plot(xx.numpy(), np.sin(xx.numpy()), "--", c = "r")
    plt.grid()
    plt.savefig("saved_file.pdf")
    plt.show()
    