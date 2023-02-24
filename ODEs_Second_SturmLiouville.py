import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Neural(nn.Module):
    def __init__(self, in_features, hid_features, out_features, total_layers, X):
        super().__init__()
        self.epoch = 0
        self.c = c
        self.eigen_hist = []
        self.loss_hist = []
        self.loss_ode_hist = []
        self.loss_trivial_hist = []
        self.search_hist = []
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.total_layers = total_layers
        self.layers = []
        self.input = nn.Linear(self.in_features, self.hid_features)
        self.eigen_layer = nn.Linear(1, 1)
        for _ in range(self.total_layers):
            self.layers.append(nn.Linear(self.hid_features, self.hid_features))
        self.output = nn.Linear(self.hid_features, self.out_features)
        self.activation = nn.Tanh()
        self.unpack(X)

    def unpack(self, X):
        self.x = X[:, 0].reshape(-1, 1)
        #self.eigen = X[:, 1].reshape(-1, 1)
        self.x.requires_grad = True
        #self.eigen.requires_grad = True

    def eval(self, x):
        eigenval = self.eigen_layer(torch.ones_like(x))
        X = torch.hstack((x, eigenval))
        X = self.input.forward(X)
        X = self.activation(X)
        for i in range(self.total_layers):
            X = self.layers[i].forward(X)
            X = self.activation(X)
        #self.eigen = X[:, 1].reshape(-1, 1).detach()
        u_pred = self.output.forward(X)
        return u_pred, eigenval

    def residual(self, x):
        u_pred, eigenval = self.eval(x)
        u = (b - x)*(x - a)*u_pred
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), retain_graph = True, create_graph = True)[0]
        res = u_xx + u*(torch.mean(eigenval**2))
        return res

    def test(self):
        u, eigenval = self.eval(self.x)
        self.ode_loss = torch.mean(self.residual(self.x)**2)
        self.trivial_loss = 1/(torch.mean(eigenval**2 + 1e-1)) + 1/(torch.mean((u**2)+1e-1))
        self.search = torch.exp(-torch.mean(torch.abs(eigenval)) + self.c)
        self.loss = self.ode_loss + self.trivial_loss + self.search
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            self.eigen_hist.append(abs(torch.mean(eigenval)))
            self.loss_hist.append(self.loss.item())
            self.loss_ode_hist.append(self.ode_loss.item())
            self.loss_trivial_hist.append(self.trivial_loss.item())
            self.search_hist.append(self.search.item())
            
            #if self.ode_loss.item() < 5e-2:
             #   torch.save(model.state_dict(), 'model.pt')
                
            if self.epoch % 1000 == 0:
                self.c += 1
                print("Loss Perturbed")
            if self.epoch % 100 == 0:
                print(f"Epoch: {self.epoch}, Total Loss: {self.loss.item():.3e}, Ode Loss: {self.ode_loss.item():.3e}, Trivial Loss: {self.trivial_loss.item():.3e}, Search Loss: {self.search.item():3e}") 
    
    def train(self):
        for _ in range(n_itter):
            self.epoch += 1
            self.test()


n_itter = 10800
steps = 1000
neurons = 12
extralayers = 0
learning_rate = 0.01
a = 0
b = 1
c = 0


fig, ax = plt.subplots(6)

epoch_hist = [x for x in range(1, n_itter + 1)]
x = torch.linspace(a, b, steps)[:, None]
eigen = torch.ones((steps, 1))
X = torch.hstack((x, eigen))

model = Neural(2, neurons, 1, extralayers, X)
model.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
model.train()

def sq_integration(x, y, a, b, points):
    with torch.no_grad():
        dx = (b - a) / (points - 1)
        k = torch.mul(y, y)
        Integral = torch.mm(k.t(), torch.ones_like(x))*dx
    return Integral.item()

#f1 = copy.deepcopy(model)

with torch.no_grad():
    y = (b - x)*(x - a)*model.eval(x)[0]
    y = y.detach()
    eigenval = model.eval(x)[1]
    eigenval = eigenval.detach()
    A = sq_integration(x, y, a, b, steps)
    y = y/np.sqrt(A)
    norm = sq_integration(x, y, a, b, steps)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("u")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Eigenvalue")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Loss_ode")
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("Loss_trivial")
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("Loss_search")
    ax[0].plot(x.numpy(), y.numpy(), c = "blue", linewidth = 2)
    ax[0].plot(x.numpy(), np.sqrt(2)*np.sin(np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[0].plot(x.numpy(), -np.sqrt(2)*np.sin(np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[0].plot(x.numpy(), np.sqrt(2)*np.sin(2*np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[0].plot(x.numpy(), -np.sqrt(2)*np.sin(2*np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[0].plot(x.numpy(), np.sqrt(2)*np.sin(3*np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[0].plot(x.numpy(), -np.sqrt(2)*np.sin(3*np.pi*x.numpy()), "--", c = "r", linewidth = 2, alpha = 0.5)
    ax[1].semilogx(np.array(epoch_hist), np.array(model.eigen_hist))
    ax[1].semilogx(np.array(epoch_hist), np.array(model.eigen_hist))
    ax[2].loglog(np.array(epoch_hist), np.array(model.loss_hist))
    ax[3].loglog(np.array(epoch_hist), np.array(model.loss_ode_hist))
    ax[4].loglog(np.array(epoch_hist), np.array(model.loss_trivial_hist))
    ax[5].loglog(np.array(epoch_hist), np.array(model.search_hist))
    print(f"Normalization: {norm:.3f}")
    print(f"Eigenvalue: {abs(torch.mean(eigenval)):.3e}")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    ax[4].grid()
    ax[5].grid()
    plt.show()
    
    # torch.save(model.state_dict(), 'model.pt')
    # state_dict = torch.load('model.pt')
    # savedmodel = Neural(2, neurons, 1, layers, X)
    # savedmodel.load_state_dict(state_dict)
    # savedode = (b - x)*(x - a)*savedmodel.eval(x)[0].detach()
    # A = sq_integration(x, savedode, a, b, steps)
    # savedode = savedode/np.sqrt(A)
    # plt.plot(x.numpy(), savedode.numpy())
    # plt.grid()
    # print(abs(torch.mean(savedmodel.eval(x)[1])))
    