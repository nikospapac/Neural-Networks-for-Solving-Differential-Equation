#%matplotlib widget
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import sys

class Neural(nn.Module):
    def __init__(self, in_features, hid_features, out_features, extra_layers, X_Boundary, X_Internal):
        super().__init__()
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.extra_layers = extra_layers
        self.epoch = 0
        
        
        str = f'nn.Linear({self.in_features}, {self.hid_features}), nn.Tanh()'
        
        for i in range(self.extra_layers):
            str += f', nn.Linear({self.hid_features}, {self.hid_features}), nn.Tanh()'
        
        str += f', nn.Linear({self.hid_features}, {self.out_features})'
        
        self.net = eval(f'nn.Sequential({str})')
        
        # self.net = nn.Sequential(nn.Linear(self.in_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.out_features))
        self.optimizer = torch.optim.LBFGS(self.parameters(), max_iter = 200, max_eval = 100)
        self.unpack(X_Boundary, X_Internal)

    def unpack(self, X_Boundary, X_Internal):
        self.t_boundary = X_Boundary[:, 0].reshape(-1, 1)
        self.x_boundary = X_Boundary[:, 1].reshape(-1, 1)
        self.t_internal = X_Internal[:, 0].reshape(-1, 1)
        self.x_internal = X_Internal[:, 1].reshape(-1, 1)
        self.t_boundary.requires_grad = True
        self.x_boundary.requires_grad = True
        self.t_internal.requires_grad = True
        self.x_internal.requires_grad = True
        
    def eval(self, t, x):
        X = torch.hstack((t, x))
        u_pred = self.net.forward(X)
        return u_pred

    def residual(self, t, x):
        u = self.eval(t, x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        #u_tt = torch.autograd.grad(u_t, y, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        res = u_xx - u_t
        return res
    
    def closure(self):
            self.optimizer.zero_grad()
            self.res_boundary = self.eval(self.t_boundary, self.x_boundary) - U_Boundary
            self.res_internal = self.residual(self.t_internal, self.x_internal)
            self.boundary_loss = torch.mean(self.res_boundary**2)
            self.internal_loss = torch.mean(self.res_internal**2)
            self.loss = 1e1*self.boundary_loss + self.internal_loss
            self.loss.backward()
            return self.loss
    """
    def test(self):
        for epoch in range(n_itter):
            #self.res_boundary = self.residual(self.x_boundary, self.y_boundary) + torch.mean((self.eval(self.x_boundary, self.y_boundary) - U_Boundary)**2)
            self.res_boundary = self.eval(self.t_boundary, self.x_boundary) - U_Boundary
            self.res_internal = self.residual(self.t_internal, self.x_internal)
            self.boundary_loss = torch.mean(self.res_boundary**2)
            self.internal_loss = torch.mean(self.res_internal**2)
            self.loss = self.boundary_loss + self.internal_loss
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Epoch: {epoch}, Total_Loss: {self.loss.item():.3e}")
            #print(f"Loss_Internal: {self.boundary_loss.item():.3e}")
            #print(f"Loss_Boundary: {self.internal_loss.item():.3e}")
            """
    def test(self):
            for epoch in range(epochs):
            #self.net.train()
                self.epoch += 1
                self.optimizer.step(self.closure)
                print(f"Epoch: {self.epoch}, Total Loss: {self.loss.item():.3e}, Boundary Loss: {self.boundary_loss.item():.3e}, Internal Loss: {self.internal_loss.item():.3e}")
   
parser = argparse.ArgumentParser()
parser.add_argument('--in_points', default = 100, type = int, help='Number of points inside the rectangle: Default 100')
parser.add_argument('--b_points', default = 100, type = int, help='Number of points in each bound: Default 100')
parser.add_argument('--neurons', default = 12, type = int, help='Number of neurons: Default 12')
parser.add_argument('--extralayers', default = 1, type = int, help='Number of extra layers: Default 1')
parser.add_argument('--epochs', default = 20, type = int, help='Epochs: Default 20')
parser.add_argument('--function', default = 'Sin', type = str, help='Available functions are: Sin or Gaussian: Default Sin')
args = parser.parse_args()
            
epochs = args.epochs
in_points = args.in_points
b_points = args.b_points
neurons = args.neurons
extra_layers = args.extralayers

default_func = ['Sin', 'Gaussian']

if args.function not in default_func:
    sys.exit("Not an available function")

fig = plt.figure()
ax = plt.axes(projection = "3d")

#index = np.arange(0, 4*b_points)
#np.random.shuffle(index)

# x --> [-1, 1]
# y --> [-1, 1]
# X --> [-1, 1]x[-1, 1]
# Generator for points in [-1, 1]
def generator(a, b, points):
    tensor = (b - a)*torch.rand((points, 1)) + a
    return tensor

t_b = generator(0, 1, b_points)
x_b = generator(-1, 1, b_points)
t_in = generator(0, 1, in_points)
x_in = generator(-1, 1, in_points)

North = torch.hstack((t_b, torch.ones_like(x_b)))
South = torch.hstack((t_b, -torch.ones_like(x_b)))
#East = torch.hstack((torch.ones_like(t_b), x_b))
West = torch.hstack((torch.zeros_like(t_b), x_b))

X_Boundary = torch.vstack((North, South, West))
#X_Boundary = X_Boundary[index, :]
X_Internal = torch.hstack((t_in, x_in))
#X = torch.vstack((X_Internal, X_Boundary))

if args.function == "Sin":
    U_Boundary = torch.vstack((torch.zeros((b_points, 1)), torch.zeros((b_points, 1)), torch.sin(x_b*torch.pi) ))
    
if args.function == "Gaussian":
    U_Boundary = torch.vstack((torch.zeros((b_points, 1)), torch.zeros((b_points, 1)),  1/np.sqrt(2*torch.pi*1e-1)*torch.exp(-x_b**2/(2*1e-1))  ))

model = Neural(2, neurons, 1, extra_layers, X_Boundary, X_Internal)
#model.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
model.test()

font = {
            "family": "calibri",
            "color": "black",
            "weight": "normal",
            "size": 13
        }

with torch.no_grad():
    t = torch.linspace(0, 1, 1000)
    x = torch.linspace(-1, 1, 1000)
    T, X = torch.meshgrid(t, x, indexing = "xy")
    Tshaped = T.reshape(-1, 1)
    Xshaped = X.reshape(-1, 1)
    u = model.eval(Tshaped, Xshaped)
    u = u.reshape(t.numel(), x.numel())
    t = t.detach().numpy()
    x = x.detach().numpy()
    u = u.detach().numpy()
    TT, XX = np.meshgrid(t, x, indexing = "xy")

    #plt.scatter(X_Boundary[:, 0].numpy(), X_Boundary[:, 1].numpy(), s = 3)
    #plt.scatter(X_Internal[:, 0].numpy(), X_Internal[:, 1].numpy(), s = 3)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("u")
    #ax.set_title(r"$u_{xx} - u_{t} = 0,$" + "\n" + r"$u(t, 1) = 0, u(t, -1) = 0, u(0, x) = sin(πx)$", fontdict = font)
    ax.plot_surface(TT, XX, u, cmap = "plasma")
    plt.grid()
    plt.show()
