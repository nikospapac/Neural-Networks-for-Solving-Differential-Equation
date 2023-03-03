import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np

x_0 = 0
y_0 = 0
y_x_0 = 0
L = 16
linspace_size_from = x_0 - L
linspace_size_to = x_0 + L
points = 1000
n_iterr = 1000
learning_rate = 0.02
neurons = 100
extralayers = 0
epoch_ls = [x for x in range(n_iterr)]
time_ls = []
loss_ls = []

t1_exec = time.time()

xx = torch.linspace(linspace_size_from, linspace_size_to, points)[:, None]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

class model(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = []
        for i in range(self.layers):
            self.hidden_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias = True))
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.activate = nn.Sigmoid()

    def forward_prop(self, x):
        x = self.input_layer.forward(x)
        x = self.activate(x)
        for i in range(0, self.layers):
            x = self.hidden_layer[i].forward(x)
            x = self.activate(x)
        x = self.output_layer.forward(x)
        return x
    
    def loss(self, x, y_pred, f, x_0, y_0, y_x_0):
        #y_derivative = torch.autograd.grad(y_0 + y_pred*(x - x_0), x, grad_outputs = torch.ones_like(x), create_graph = True)
        y = y_0 + y_pred*(x - x_0)
        #y_pred_x = self.derivative(x, y_pred, 1)
        #self.y_pred_x = y_pred_x[0].detach()
        y_x = self.derivative(x, y, 1)
        #y_xx = self.derivative(x, y, 2)
        loss = torch.mean((y_x[0] - torch.cos(x) - f).pow(2))
        loss.backward()
        return loss
    
    def derivative(self, x, y, deriv_order):
        derivative = y
        for order in range(deriv_order):
            derivative = torch.autograd.grad(derivative, x, grad_outputs = torch.ones_like(x), create_graph = True)
        return derivative

model = model(1, neurons, 1, extralayers)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(0, n_iterr):
        t1_train = time.time()
        x = 2*L*torch.rand((points, 1), requires_grad = True) - L
        y_pred = model.forward_prop(x)
        loss = model.loss(x, y_pred, 0, x_0, y_0, y_x_0)
        optimizer.step()
        optimizer.zero_grad()
        print("Loss: ", loss.item())
        t2_train = time.time()
        time_ls.append(t2_train - t1_train)
        loss_ls.append(loss.item())

ax2.plot(epoch_ls, time_ls)
ax3.semilogy(epoch_ls, loss_ls)
y = y_0 + y_pred*(x - x_0)
x.detach_()
y.detach_()
with torch.no_grad():
    yy = y_0 + model.forward_prop(xx)*(xx - x_0)
    zz = model.forward_prop(xx)
ax1.scatter(x.numpy(), y.numpy(), s = 3, c = "red")
ax1.plot(xx.numpy(), np.sin(xx.numpy()))
##plt.plot(xx.numpy(), yy.numpy())
#plt.plot(xx.numpy(), zz.numpy())
ax1.grid()
ax1.set_ylabel("y")
ax1.set_xlabel("x")
ax2.grid()
ax2.set_ylabel("Time (sec)")
ax2.set_xlabel("Epoch")
ax3.grid()
ax3.set_ylabel("Loss")
ax3.set_xlabel("Epoch")
#plt.show()
t2_exec = time.time()
#fig1.savefig("saved_file.pdf")
print("Execution Time: ", t2_exec - t1_exec)



