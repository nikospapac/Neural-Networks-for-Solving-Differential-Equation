import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import imageio

a = -10
b = 10
points = 1000
neurons = 100
learning_rate = 0.1
epochs = 1000
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
        self.optimizer.zero_grad()
        self.epoch += 1
        self.y_pred = self.forward(self.x)
        self.y = torch.exp(-self.x**2)
        self.loss = torch.mean((self.y - self.y_pred)**2)
        print(f"[{self.epoch}]Loss: {self.loss.item():.3e}")
        self.loss.backward()
        return self.loss
    
    def run(self):
        for i in range(0, epochs + 1):
            if i % 10 == 0:
                with torch.no_grad():
                    plt.clf()
                    y_pred = fourier.forward(torch.Tensor(x))
                    y = np.exp(-x**2)
                    plt.plot(x, np.array(y_pred))
                    plt.plot(x, np.array(y), "--")
                    plt.xlim(a, b)
                    plt.ylim(0, 2)
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.grid()
                    plt.savefig(f"images\{i}.pdf")
                    
            self.optimizer.step(self.closure)


fourier = net(x, 1, neurons, 1)
fourier.run()


with torch.no_grad():
    plt.clf()
    y_pred = fourier.forward(torch.Tensor(x))
    y = np.exp(-x**2)
    error = np.abs(y_pred - y)
    plt.plot(x, error)
    plt.xlim(a, b)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid()
    plt.show()
    fig.savefig(f"errorimages\error.pdf")
    


# files = sorted(os.listdir("images"), key = len)
# image_path = [os.path.join("images", file) for file in files]
# images = []

# for img in image_path:
#     images.append(imageio.imread(img))

# imageio.mimwrite("output/animation.gif", images, fps = 5)


# k1 = fourier.phase.weight[0].item()
# f1 = fourier.phase.bias[0].item()
# k2 = fourier.phase.weight[1].item()
# f2 = fourier.phase.bias[1].item()
# do = fourier.out.bias[0].item()
# d1 = fourier.out.weight[0, 0].item()
# d2 = fourier.out.weight[0, 1].item()

# print(f"y = {do:.2f} + {d1:.2f}sin({k1:.2f}x + {f1:.2f}) + {d2:.2f}sin({k2:.2f}x + {f2})")
# eq = do + d1*np.sin(k1*x+f1) + d2*np.sin(k2*x+f2)


# with torch.no_grad():
#     y_pred = fourier.forward(torch.Tensor(x))
#     y = np.exp(-x**2)
#     plt.plot(x, np.array(y_pred))
#     plt.plot(x, np.array(y), "--")
#     plt.grid()
#     plt.show()