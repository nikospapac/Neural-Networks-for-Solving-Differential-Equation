import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


linspace_size_from = -5
linspace_size_to = 5
steps = 1000
dx = (linspace_size_to - linspace_size_from) / (steps - 1) #I use linspace so, dx is the same for every step
n_iter = 200
learning_rate = 0.1
neurons = 10
extralayers = 0
function_name = r"$y'(x) = cos(x), y(0) = 0$"
partitions = 21 #partitions must be an odd number
x_0 = 0
y_0 = 0
y_pred_R = []
y_pred_L = []



class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super().__init__()
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
    
    def loss(self, x, y_pred, f, x_0, y_0):
        #y_derivative = torch.autograd.grad(y_0 + y_pred*(x - x_0), x, grad_outputs = torch.ones_like(x), create_graph = True)
        y = y_0 + y_pred*(x - x_0)
        y_x = self.derivative(x, y, 1)
        loss = torch.mean((y_x[0] - f).pow(2))
        loss.backward()
        return loss

    def derivative(self, x, y, deriv_order):
        derivative = y
        for order in range(deriv_order):
            derivative = torch.autograd.grad(derivative, x, grad_outputs = torch.ones_like(x), create_graph = True)
        return derivative



for j in range(0, 1):
    x_linspace = torch.linspace(linspace_size_from, linspace_size_to, steps)[:,None]
    chunk_size = int(len(x_linspace)/partitions)
    x = torch.split(x_linspace, chunk_size)
    x_linspace = x_linspace.numpy()

    font = {
                "family": "calibri",
                "color": "black",
                "weight": "normal",
                "size": 13
            }
    
    def f(x):
        return f"torch.cos({x})"
    
    
    
    def initializer(x, x_0):
        chunk_index = chunk_find(x, x_0)
        model = net(1, neurons, 1, extralayers)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        x_R = x[chunk_index:]
        x_L = x[:chunk_index]
        y_pred_list_R = [0 for n in range(len(x_R))]
        y_pred_list_L = [0 for n in range(len(x_L))]
        loss_list_R = [0 for n in range(len(x_R))]
        loss_list_L = [0 for n in range(len(x_L))]
        return model, optimizer, x_R, x_L, y_pred_list_R, y_pred_list_L, loss_list_R, loss_list_L, chunk_index

    def chunk_find(x, x_0):  #Find the index of the chunk where x_0 lays
        chunk_index = 0
        chunk_found = False
        for chunk in x:
                if (chunk.min() <= x_0 <= chunk.max()):
                    chunk_found = True
                    break
                if abs(x_0 - chunk.min()) <= dx/2 or abs(x_0 - chunk.max()) < dx/2:
                    chunk_found = True
                    break
                chunk_index += 1
        return chunk_index

    def learning(x, f, x_0, y_0):
        x.requires_grad_()
        for epoch in range(n_iter):
            y_pred = model.forward_prop(x)
            loss = model.loss(x, y_pred, f, x_0, y_0)
            optimizer.step()
            optimizer.zero_grad()
        x.requires_grad = False
        y_pred = y_pred.detach()
        return y_pred, loss.item()


    model, optimizer, x_R, x_L, y_pred_list_R, y_pred_list_L, loss_list_R, loss_list_L, chunk_index = initializer(x, x_0)

    init_x_0 = x_0
    init_y_0 = y_0

    for partition_index in range(len(x_R)):
        y_pred_list_R[partition_index], loss_list_R[partition_index] = learning(x_R[partition_index],
                                                                                eval(f("x_R[partition_index]")), 
                                                                                init_x_0, 
                                                                                init_y_0)
        plt.scatter(x_R[partition_index].numpy(), init_y_0 + (x_R[partition_index].numpy() - init_x_0)*y_pred_list_R[partition_index].numpy(), s = 3)
        init_y_0 = (init_y_0 + (x_R[partition_index][-1] - init_x_0)*y_pred_list_R[partition_index][-1]).item()
        init_x_0 = x_R[partition_index][-1].item()



    init_y_0 = (y_0 + (x_R[0][0] - x_0)*y_pred_list_R[0][0]).item()
    init_x_0 = x_R[0][0].item()

    for partition_index in range(len(x_L)):
        reversed_partition_index = len(x_L) - partition_index - 1
        y_pred_list_L[reversed_partition_index], loss_list_L[reversed_partition_index] = learning(x_L[reversed_partition_index], 
                                                                                                  eval(f("x_L[reversed_partition_index]")), 
                                                                                                  init_x_0, 
                                                                                                  init_y_0)                                                                  
        plt.scatter(x_L[reversed_partition_index].numpy(), (init_y_0 + (x_L[reversed_partition_index] - init_x_0)*y_pred_list_L[reversed_partition_index]).numpy(), s = 3)
        init_y_0 = (init_y_0 + (x_L[reversed_partition_index][0] - init_x_0)*y_pred_list_L[reversed_partition_index][0])[0].item()
        init_x_0 = x_L[reversed_partition_index][0].item()


    for i in range(0, len(x)):
        if i < len(x_L):
            print(f"Loss[{x_L[i].min():.5f}, {x_L[i].max():.5f}]: {loss_list_L[i]:.5f}")
        else:
            print(f"Loss[{x_R[i - len(x_L)].min():.5f}, {x_R[i - len(x_L)].max():.5f}]: {loss_list_R[i - len(x_L)]:.5f}")


    #plt.scatter(x_linspace, x_linspace*y_pred, s = 3)
    plt.plot(x_linspace, np.sin(x_linspace), "r")
    #plt.title(function_name + f"\n Chunk Size: {chunk_size}, Partitions: {partitions}, Neurons: {neurons}, Layers: {extralayers}", fontdict = font)
    plt.grid()
    plt.xlabel("x", fontdict = font)
    plt.ylabel("y", fontdict = font)
    #plt.savefig(f"fig_{function_name}_{partitions}.jpeg")
    plt.show()
    
    partitions += 2
