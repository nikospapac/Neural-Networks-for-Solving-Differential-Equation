import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time

x_0 = 0
y_0 = 1
y_x_0 = 0
linspace_size_from = x_0 - 4
linspace_size_to = 4 + x_0
steps = 1000
#I use linspace so, dx is the same for every step
dx = (linspace_size_to - linspace_size_from) / (steps - 1)
n_iter = 3000
learning_rate = 0.01
neurons = 100
extralayers = 0
function_name = r"$y''(x) = -sin(y(x)), y(0) = 1, y'(0) = 0$"
partitions = 1 #partitions must be an odd number


class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias = False)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activate = nn.Sigmoid()


    def forward_prop(self, x):
        x = self.input_layer.forward(x)
        x = self.activate(x)
        x = self.output_layer.forward(x)
        return x
    
    def loss(self, x, y_pred, f, x_0, y_0, y_x_0):
        #y_derivative = torch.autograd.grad(y_0 + y_pred*(x - x_0), x, grad_outputs = torch.ones_like(x), create_graph = True)
        y = y_0 + y_x_0*(x - x_0) + y_pred*(x - x_0)**2
        y_pred_x = self.derivative(x, y_pred, 1)
        self.y_pred_x = y_pred_x[0].detach()
        y_x = self.derivative(x, y, 1)
        y_xx = self.derivative(x, y, 2) 
        loss = torch.mean((y_xx[0] + y - f).pow(2))
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
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    def f(x):
        return f"{0}"
    
    
    
    def initializer(x, x_0):
        
        chunk_index = chunk_find(x, x_0)
        model = net(1, neurons, 1, extralayers)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        x_R = x[chunk_index:]
        x_L = x[:chunk_index]
        y_pred_list_R = [0 for n in range(len(x_R))]
        y_pred_list_L = [0 for n in range(len(x_L))]
        y_x_pred_list_R = [0 for n in range(len(x_R))]
        y_x_pred_list_L = [0 for n in range(len(x_L))]
        loss_list_R = [0 for n in range(len(x_R))]
        loss_list_L = [0 for n in range(len(x_L))]
        
        return (model, optimizer, x_R, x_L, y_pred_list_R, 
                y_pred_list_L, y_x_pred_list_R, y_x_pred_list_L, 
                loss_list_R, loss_list_L, chunk_index)

    def chunk_find(x, x_0):  #Find the index of the chunk where x_0 lays
        
        chunk_index = 0
        chunk_found = False
        for chunk in x:
                if (chunk.min() <= x_0 <= chunk.max()):
                    chunk_found = True
                    break
                chunk_index += 1
                
        return chunk_index

    def learning(x, f, x_0, y_0, y_x_0):
        x.requires_grad_()
        
        for epoch in range(n_iter):
            
            y_pred = model.forward_prop(x)
            loss = model.loss(x, y_pred, f, x_0, y_0, y_x_0)
            optimizer.step()
            optimizer.zero_grad()
            
        x.requires_grad = False
        y_pred = y_pred.detach()
        y_pred_x = model.y_pred_x
        return y_pred, y_pred_x, loss.item()


    (model, optimizer, x_R, x_L, y_pred_list_R, 
    y_pred_list_L, y_x_pred_list_R,
    y_x_pred_list_L, loss_list_R, loss_list_L, chunk_index) = initializer(x, x_0)



    init_x_0 = x_0
    init_y_0 = y_0
    init_y_x_0 = y_x_0
    
    t1 = time.time()
    for partition_index in range(len(x_R)):
        
        (y_pred_list_R[partition_index], y_x_pred_list_R[partition_index], 
        loss_list_R[partition_index]) = learning(x_R[partition_index],
                                                eval(f("x_R[partition_index]")), 
                                                init_x_0, 
                                                init_y_0,
                                                init_y_x_0)
        
        y_R = (init_y_0 + (x_R[partition_index] - init_x_0)*init_y_x_0 +
                y_pred_list_R[partition_index]*(x_R[partition_index] - init_x_0)**2)
        
        y_x_R = (init_y_x_0 + 2*(x_R[partition_index] - init_x_0)*y_pred_list_R[partition_index] +
                y_x_pred_list_R[partition_index]*(x_R[partition_index] - init_x_0)**2)
        
        ax1.plot(x_R[partition_index].numpy(), y_R.numpy())
        ax2.plot(y_R.numpy(), y_x_R.numpy())
        
        init_y_0 = (init_y_0 + (x_R[partition_index][-1] - init_x_0)*init_y_x_0 +
                    y_pred_list_R[partition_index][-1]*(x_R[partition_index][-1] - init_x_0)**2).item()
        
        init_y_x_0 = (init_y_x_0 + 2*(x_R[partition_index][-1] - init_x_0)*y_pred_list_R[partition_index][-1] +
                     y_x_pred_list_R[partition_index][-1]*(x_R[partition_index][-1] - init_x_0)**2).item()
        
        init_x_0 = x_R[partition_index][-1].item()
    t2 = time.time()
    
    print("dt_right = ", t2-t1)
    t3 = time.time()
    init_x_0 = x_R[0][0].item()
    init_y_0 = (y_0 + y_x_0*(x_R[0][0] - x_0) + y_pred_list_R[0][0]*(x_R[0][0] - x_0)**2).item()
    init_y_x_0 = (y_x_0 + 2*(x_R[0][0] - x_0)*y_pred_list_R[0][0] +
                 y_x_pred_list_R[0][0]*(x_R[0][0] - x_0)**2).item()


    for partition_index in range(len(x_L)):
        
        reversed_partition_index = len(x_L) - partition_index - 1
        
        (y_pred_list_L[reversed_partition_index], y_x_pred_list_L[reversed_partition_index],
        loss_list_L[reversed_partition_index]) = learning(x_L[reversed_partition_index],
                                                        eval(f("x_L[reversed_partition_index]")), 
                                                        init_x_0,
                                                        init_y_0,
                                                        init_y_x_0)
        
        y_L = (init_y_0 + (x_L[reversed_partition_index] - init_x_0)*init_y_x_0 + 
               y_pred_list_L[reversed_partition_index]*(x_L[reversed_partition_index] - init_x_0)**2)
        
        y_x_L = (init_y_x_0 + 2*(x_L[reversed_partition_index] - init_x_0)*y_pred_list_L[reversed_partition_index] +
                y_x_pred_list_L[reversed_partition_index]*(x_L[reversed_partition_index] - init_x_0)**2)
        
        ax1.plot(x_L[reversed_partition_index].numpy(), y_L.numpy())
        ax2.plot(y_L.numpy(), y_x_L.numpy())
        
        init_y_0 = (init_y_0 + (x_L[reversed_partition_index][0] - init_x_0)*init_y_x_0 +
                    y_pred_list_L[reversed_partition_index][0]*(x_L[reversed_partition_index][0] - init_x_0)**2)
        
        init_y_x_0 = (init_y_x_0 + 2*(x_L[reversed_partition_index][0] - init_x_0)*y_pred_list_L[reversed_partition_index][0] +
                    y_x_pred_list_L[reversed_partition_index][0]*(x_L[reversed_partition_index][0] - init_x_0)**2)                 
                                                  
        init_x_0 = x_L[reversed_partition_index][0]
    t4 = time.time()
    print("dt_left = ", t4-t3)
    with open(f'Compiler_Parameters.txt', "a") as fopen:
        fopen.write(f'Differential Equation: y\' = x\nlinspace[{linspace_size_from}, {linspace_size_to}]\n\
        steps: {steps}\nn_iter = {n_iter}\nlearning_rate: {learning_rate}\n\
        hidden_layers: {neurons}\nhidden_size: {neurons}\ndx: {dx}\n\n\n')


    for i in range(0, len(x)):
        
        if i < len(x_L):
            print(f"Loss[{x_L[i].min():.5f}, {x_L[i].max():.5f}]: {loss_list_L[i]}")
        else:
            print(f"Loss[{x_R[i - len(x_L)].min():.5f}, {x_R[i - len(x_L)].max():.5f}]: {loss_list_R[i - len(x_L)]}")


    #plt.scatter(x_linspace, x_linspace*y_pred, s = 3)
    #ax1.plot(x_linspace, (1/np.sqrt(5))*np.sin(np.sqrt(5)*x_linspace), "r")
    #ax1.plot(x_linspace, (2/np.sqrt(19))*np.exp(-x_linspace*0.5)*np.sin((np.sqrt(19)/2)*x_linspace - x_0), "r")
    ax1.set_title(function_name +
                  f"\n Chunk Size: {chunk_size}, Partitions: {partitions},Neurons: {neurons}, Layers: {extralayers}",
                  fontdict = font)

    ax1.grid()
    
    ax1.set_xlabel("x", fontdict = font)
    ax1.set_ylabel("y", fontdict = font)
    ax2.set_title(function_name + 
                  f"\n Chunk Size: {chunk_size}, Partitions: {partitions}, Neurons: {neurons}, Layers: {extralayers}",
                  fontdict = font)
    
    ax2.grid()
    
    ax2.set_xlabel("y", fontdict = font)
    ax2.set_ylabel("y'", fontdict = font)
    # fig1.savefig(f"fig_{function_name}_{partitions}.jpeg")
    # fig2.savefig(f"Phase_Space_fig_{function_name}_{partitions}.jpeg")
    
    
    #plt.show()
    
    partitions += 2


x0 = 0
y0 = 1
yx0 = 0
solforw = []
solbackw = []
xxforw = []
xxbackw = []
a = -4
b = 4
n = steps
dx = (b - a)/(2*(n-1))
a = 1
#y'' = cosy

def f(x, y, parameter):
    if parameter == "alpha":
        return -np.sin(y)
    if parameter == "beta":
        return y


def rk4(x, y, parameter):
    k1 = f(x, y, parameter)
    k2 = f(x + dx/2, y + dx*k1/2, parameter)
    k3 = f(x + dx/2, y + dx*k2/2, parameter)
    k4 = f(x + dx, y + dx*k3, parameter)
    return dx/6*(k1 + 2*k2 + 2*k3 + k4)

y = y0
x = x0
u1 = y0
u2 = yx0

for i in range(n):
    solforw.append(y)
    xxforw.append(x)
    u2 = u2 + rk4(x, u1, parameter = "alpha")
    u1 = u1 + rk4(x, u2, parameter = "beta")
    y = u1
    x = x + dx

y = y0
x = x0
u1 = y0
u2 = yx0

for i in range(n):
    solbackw.append(y)
    xxbackw.append(x)
    u2 = u2 - rk4(x, u1, parameter = "alpha")
    u1 = u1 - rk4(x, u2, parameter = "beta")
    y = u1
    x = x - dx
    
    
ax1.plot(np.array(xxforw), np.array(solforw), "--", color = "blue")
ax1.plot(np.array(xxbackw), np.array(solbackw), "--", color = "blue")

fig1.savefig(f"fig_{function_name}_{partitions}.pdf")
fig2.savefig(f"Phase_Space_fig_{function_name}_{partitions}.pdf")

plt.show()