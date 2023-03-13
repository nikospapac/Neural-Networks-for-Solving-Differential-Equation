import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', default = -5, type = float, help='Left bound: Default -5')
parser.add_argument('-b', default = 5, type = float, help='Right bound: Default 5')
parser.add_argument('--points', default = 100, type = int, help='Number of points: Default 100')
parser.add_argument('--neurons', default = 100, type = int, help='Number of neurons: Default 100')
parser.add_argument('--learning_rate', default = 1, type = float, help='Learning Rate: Default 1')
parser.add_argument('--epochs', default = 1000, type = int, help='Epochs: Default 1000')
parser.add_argument('--function', default = 'Gaussian', type = str, help='Available functions are: Gaussian or SquarePulse: Default Gaussian')
args = parser.parse_args()


a = args.a
b = args.b
points = args.points
neurons = args.neurons
learning_rate = args.learning_rate
epochs = args.epochs
default_func = ['Gaussian', 'SquarePulse']

if args.function not in default_func:
    sys.exit("Not an available function")
    

    
    

x = np.linspace(a, b, points)[:, None]

fig, ax = plt.subplots()



class net(nn.Module):
    def __init__(self, x, in_features, hid_features, out_features):
        super().__init__()
        self.x = torch.Tensor(x)
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.epoch = 0
        self.phase = nn.Linear(self.in_features, self.hid_features)
        self.out = nn.Linear(self.hid_features, self.out_features)
        # self.optimizer = torch.optim.LBFGS(self.parameters(),
        #                             lr = 1,
        #                             max_iter = 100,
        #                             max_eval = 100,
        #                             history_size = 100,
        #                             tolerance_grad = 1e-9,
        #                             tolerance_change = 1e-9,
        #                             line_search_fn = "strong_wolfe")
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        
    def forward(self, x):
        x = self.phase(x)
        x = torch.cos(x)
        y_pred = self.out(x)
        return y_pred
    
    
    
    def closure(self):
        if args.function == 'Gaussian':
            self.y = torch.exp(-self.x**2)
        else:
            self.y = torch.heaviside(1-self.x**2, torch.tensor([0.5]))
        self.optimizer.zero_grad()
        self.epoch += 1
        self.y_pred = self.forward(self.x)
        #self.y = torch.exp(-self.x**2)
        #self.y = torch.heaviside(1-self.x**2, torch.tensor([0.5]))
        self.loss = torch.mean((self.y - self.y_pred)**2)
        print(f"[{self.epoch}]Loss: {self.loss.item():.3e}")
        self.loss.backward()
        return self.loss
    
    def run(self):
        for i in range(0, epochs + 1):
            if i % 10 == 0:
                with torch.no_grad():
                    plt.clf()
                    xx = (2*(b-a)*torch.rand(points, requires_grad = True) - (b-a))[:, None]
                    xx.requires_grad = True
                    y_pred = fourier.forward(torch.Tensor(xx))
                    #y = np.exp(-x**2)
                    if args.function == 'Gaussian':
                        y = np.exp(-x**2)
                    else:
                        y = np.heaviside(1-x**2, 0.5)
                    y_show = fourier.forward(torch.Tensor(x))
                    plt.plot(x, np.array(y_show))
                    plt.plot(x, y, "--")
                    plt.xlim(a, b)
                    plt.ylim(0, 2)
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.grid()
                    
                      
            self.optimizer.step(self.closure)
        plt.show()


fourier = net(x, 1, neurons, 1)
fourier.run()


with torch.no_grad():
    plt.clf()
    y_pred = fourier.forward(torch.Tensor(x))
    if args.function == 'Gaussian':
        y = np.exp(-x**2)
    else:
        y = np.heaviside(1-x**2, 0.5)
    error = np.abs(y_pred - y)
    plt.plot(x, error)
    plt.xlim(a, b)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid()
    plt.show()
