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
        # self.net = nn.Sequential(nn.Linear(self.in_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.hid_features),
        #                          nn.Tanh(),
        #                          nn.Linear(self.hid_features, self.out_features))
        
        str = f'nn.Linear({self.in_features}, {self.hid_features}), nn.Tanh()'
        
        for i in range(self.extra_layers):
            str += f', nn.Linear({self.hid_features}, {self.hid_features}), nn.Tanh()'
        
        str += f', nn.Linear({self.hid_features}, {self.out_features})'
        
        self.net = eval(f'nn.Sequential({str})')
        self.optimizer = torch.optim.LBFGS(self.parameters())
        # self.optimizer = torch.optim.LBFGS(self.parameters(),
        #                             lr = 1,
        #                             max_iter = 1000,
        #                             max_eval = 1000,
        #                             history_size = 100,
        #                             tolerance_grad = 1e-9,
        #                             tolerance_change = 0.5 * np.finfo(float).eps,
        #                             line_search_fn = "strong_wolfe")
        self.unpack(X_Boundary, X_Internal)
        
    def unpack(self, X_Boundary, X_Internal):
        self.x_boundary = X_Boundary[:, 0].reshape(-1, 1)
        self.y_boundary = X_Boundary[:, 1].reshape(-1, 1)
        self.x_internal = X_Internal[:, 0].reshape(-1, 1)
        self.y_internal = X_Internal[:, 1].reshape(-1, 1)
        self.x_boundary.requires_grad = True
        self.y_boundary.requires_grad = True
        self.x_internal.requires_grad = True
        self.y_internal.requires_grad = True
        
    def eval(self, x, y):
        X = torch.hstack((x, y))
        u_pred = self.net.forward(X)
        return u_pred
    
    def residual(self, x, y):
        u = self.eval(x, y)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        if args.function == 'Poisson':
            res = u_xx + u_yy + 1/np.sqrt(1*np.pi*1e-1)*torch.exp(-0.5*1e1*(x**2 + y**2)) #poisson
        if args.function == 'Laplace':
            res = u_xx + u_yy
        return res
    
    def closure(self):
        self.optimizer.zero_grad()
        self.res_boundary = self.eval(self.x_boundary, self.y_boundary) - U_Boundary
        self.res_internal = self.residual(self.x_internal, self.y_internal)
        self.boundary_loss = torch.mean(self.res_boundary**2)
        self.internal_loss = torch.mean(self.res_internal**2)
        self.loss = self.boundary_loss + 0.1*self.internal_loss
        self.loss.backward()
        return self.loss
    
    def test(self):
        for _ in range(epochs):
        #self.net.train()
            self.epoch += 1
            self.optimizer.step(self.closure)
            print(f"Epoch: {self.epoch}, Total Loss: {self.loss.item():.3e}, Boundary Loss: {self.boundary_loss.item():.3e}, Internal Loss: {self.internal_loss.item():.3e}")
        
parser = argparse.ArgumentParser()
parser.add_argument('--in_points', default = 100, type = int, help='Number of points inside the rectangle')
parser.add_argument('--b_points', default = 100, type = int, help='Number of points in each bound')
parser.add_argument('--neurons', default = 12, type = int, help='Number of neurons')
parser.add_argument('--extralayers', default = 2, type = int, help='Number of extra layers')
parser.add_argument('--epochs', default = 50, type = int, help='Epochs')
parser.add_argument('--function', default = 'Laplace', type = str, help='Available functions are: Poisson or Laplace')
args = parser.parse_args()
        
epochs = args.epochs
in_points = args.in_points
b_points = args.b_points
neurons = args.neurons
extralayers = args.extralayers
default_func = ['Poisson', 'Laplace']

if args.function not in default_func:
    sys.exit("Not an available function")

fig = plt.figure()
ax = plt.axes(projection = "3d")

fig2d, ax2d = plt.subplots()

# x --> [-1, 1]
# y --> [-1, 1]
# X --> [-1, 1]x[-1, 1]
# Generator for points in [-1, 1]

def generator(a, b, points):
    tensor = (b - a)*torch.rand((points, 1)) + a
    return tensor

x_b = generator(-1, 1, b_points)
y_b = generator(-1, 1, b_points)
x_in = generator(-1, 1, in_points)
y_in = generator(-1, 1, in_points)

#Boundary Coordinates
North = torch.hstack((x_b, 1*torch.ones_like(y_b)))
South = torch.hstack((x_b, -1*torch.ones_like(y_b)))
East = torch.hstack((1*torch.ones_like(x_b), y_b))
West = torch.hstack((-1*torch.ones_like(x_b), y_b))

X_Boundary = torch.vstack((North, South, East, West))
X_Internal = torch.hstack((x_in, y_in))
model = Neural(2, neurons, 1, extralayers, X_Boundary, X_Internal)
#U_Boundary = torch.vstack(( torch.ones((b_points, 1)) , torch.ones((b_points, 1)), torch.zeros((b_points, 1)), torch.zeros((b_points, 1)) )) #Boundary Counditions (North, South, East, West)
#U_Boundary = torch.vstack(( torch.sin(x_b*torch.pi) , -torch.sin(x_b*torch.pi), -torch.sin(y_b*torch.pi), torch.sin(y_b*torch.pi) ))

if args.function == "Poisson":
    U_Boundary = torch.vstack(( torch.zeros_like(x_b) , torch.zeros_like(x_b), torch.zeros_like(y_b), torch.zeros_like(y_b) )) #NSEW

if args.function == "Laplace":
    U_Boundary = torch.vstack(( torch.zeros_like(x_b) , torch.zeros_like(x_b), 1 - y_b**2, torch.zeros_like(y_b) )) #NSEW

model.test()

font = {
            "family": "calibri",
            "color": "black",
            "weight": "normal",
            "size": 13
        }

with torch.no_grad():
    x = torch.linspace(-1, 1, 1000)
    y = torch.linspace(-1, 1, 1000)
    X, Y = torch.meshgrid(x, y, indexing = "xy")
    Xshaped = X.reshape(-1, 1)
    Yshaped = Y.reshape(-1, 1)
    u = model.eval(Xshaped, Yshaped)
    u = u.reshape(x.numel(), y.numel())
    x = x.detach().numpy()
    y = y.detach().numpy()
    u = u.detach().numpy()
    XX, YY = np.meshgrid(x, y, indexing = "xy")

    #plt.scatter(X_Boundary[:, 0].numpy(), X_Boundary[:, 1].numpy(), s = 3)
    #plt.scatter(X_Internal[:, 0].numpy(), X_Internal[:, 1].numpy(), s = 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    #ax.set_title(r"$u_{xx} + u_{yy} = -\frac{1}{\sqrt{2πσ^2}}Exp(-\frac{x^2 + y^2}{2σ^2}), σ^2 = 0.1,$" + "\n" + r"$u(x, 1) = 0, u(x, -1) = 0, u(-1, y) = 0, u(1, y) = 0$", fontdict = font)
    ax.plot_surface(XX, YY, u, cmap = "plasma")
    ax2d.contourf(XX, YY, u, cmap = "plasma")
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax.grid()
    plt.show()