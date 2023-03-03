import numpy as np
import matplotlib.pyplot as plt


def solve(yo, yxo, eigenval):
    y = yo
    yx = yxo
    N = 400
    a = 0
    b = 1
    yls = [yo]
    yxls = [yxo]
    x = np.linspace(a, b, N)
    dx = (b - a)/(N - 1)

    def f(eigenval, y):
        return -y*eigenval**2

    for _ in range(N - 1):
        y += dx*yx
        yx += dx*f(eigenval, y)
        
        yxls.append(yx)
        yls.append(y)

    return x, yls, yls[-1]

dl = 0.001
lo = np.linspace(1, 20, 10)
l = lo
llist = []

def f(l):
    return solve(0, 1, l)[2]

def dfdl(l, dl):
    return (f(l+dl) - f(l))/dl

def newtonraphson(l, dl, N):
    for _ in range(N):
            l += -f(l)/dfdl(l, dl)
    return l

for ls in l:
    llist.append(newtonraphson(ls, dl, 20))



def datacleaner(list, threshhold):
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            if abs(list[i] - list[j]) < threshhold:
                list[j] = 0
    newlistlist = [x for x in list if x != 0]
    newlistlist.sort()
    return newlistlist

#print(llist)
llist = datacleaner(llist, 0.001)
print("Eigenvalues: ", llist)

n = 3 # Choose the eigenfunction you want to show
l = llist[n - 1]
x, y, _ =  solve(0, 1, l)

plt.plot(x, y)
plt.plot(x, np.sin(n*np.pi*x)/(n*np.pi), "--")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.savefig("eigenshooting.pdf")
plt.show()