import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


x_1 = -1
y_1 = 0
x_2 = 1
y_2 = 0
linspace_size_from = x_1
linspace_size_to = x_2
steps = 1000
dx = (linspace_size_to - linspace_size_from) / (steps - 1) #I use linspace so, dx is the same for every step
n_iter = 1000
learning_rate = 0.1
neurons = 5
extralayers = 0
function_name = r"$y''(x) = x, y(-1) = 0, y(1) = 0$"



class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, x_1, y_1, x_2, y_2):
        super().__init__()
        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2
        self.hidden_layer = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        for i in range(self.hidden_layers):
            self.hidden_layer.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activate = nn.Sigmoid()


    def forward_prop(self, x):
        x = self.input_layer.forward(x)
        x = self.activate(x)
        for i in range(self.hidden_layers - 1):
            x = self.hidden_layer[i].forward(x)
            x = self.activate(x)
        x = self.output_layer.forward(x)
        return x
    
    def loss(self, x, y_pred, f):
        L = self.x_2 - self.x_1
        y = self.y_2*(x - self.x_1)/L + self.y_1*(self.x_2 - x)/L + (x - self.x_1)*(x - self.x_2)*y_pred
        y_x = self.derivative(x, y, 1)
        y_xx = self.derivative(x, y, 2)
        loss = torch.mean((y_xx[0] - f).pow(2)) 
        loss.backward()
        return loss

    def derivative(self, x, y, deriv_order):
        derivative = y
        for order in range(deriv_order):
            derivative = torch.autograd.grad(derivative, x, grad_outputs = torch.ones_like(x), create_graph = True)
        return derivative

x_linspace = torch.linspace(linspace_size_from, linspace_size_to, steps)[:,None]
x = x_linspace
x_linspace = x_linspace.numpy()

font = {
            "family": "calibri",
            "color": "black",
            "weight": "normal",
            "size": 13
        }

def initializer(x_1, y_1, x_2, y_2):
    model = net(1, neurons, 1, extralayers, x_1, y_1, x_2, y_2)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    return model, optimizer

def learning(x, f):
    x.requires_grad_()
    for epoch in range(n_iter):
        y_pred = model.forward_prop(x)
        loss = model.loss(x, y_pred, f)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss[{epoch}]: {loss.item()}")
    x.requires_grad = False
    y_pred = y_pred.detach().numpy()
    loss = loss.detach()
    return y_pred, loss

def B(x, x_1, y_1, x_2, y_2):
    L = x_2 - x_1
    B_val = y_2*(x - x_1)/L + y_1*(x_2 - x)/L
    return B_val

model, optimizer = initializer(x_1, y_1, x_2, y_2)
y_pred, loss = learning(x, x)
print("Loss: ", loss.item())


with open(f'Compiler_Parameters.txt', "a") as fopen:
    fopen.write(f'Differential Equation: y\' = x\nlinspace[{linspace_size_from}, {linspace_size_to}]\n\
    steps: {steps}\nn_iter = {n_iter}\nlearning_rate: {learning_rate}\n\
    hidden_layers: {neurons}\nhidden_size: {neurons}\ndx: {dx}\n\n\n')

y = B(x_linspace, x_1, y_1, x_2, y_2) + (x_linspace-x_1)*(x_linspace-x_2)*y_pred

plt.scatter(x_linspace, y, s = 3)
plt.plot(x_linspace, (x_linspace**3 - x_linspace)/6, "r")
plt.grid()
plt.title(function_name + f"\nNeurons: {neurons}, Layers: {extralayers}", fontdict = font)
plt.xlabel("x", fontdict = font)
plt.ylabel("y", fontdict = font)
plt.savefig(f"fig_{function_name}.pdf")
plt.show()
