import numpy as np
import matplotlib.pyplot as plt

L = 1
M = 5 #number of non overlapping triangle i want
elements = 2*M-1
N = 2*M + 1 # points
h = L/N
Nsample = 500
hsample = L/Nsample
xgrid = np.linspace(-L, L, N)
xsample = np.linspace(-L, L, Nsample)

# A = 1/h*np.array([[ 2, -1,  0,  0,  0,  0,  0,  0,  0], 
#                   [-1,  2, -1,  0,  0,  0,  0,  0,  0], 
#                   [ 0, -1,  2, -1,  0,  0,  0,  0,  0],
#                   [ 0,  0, -1,  2, -1,  0,  0,  0,  0],
#                   [ 0,  0,  0, -1,  2, -1,  0,  0,  0],
#                   [ 0,  0,  0,  0, -1,  2, -1,  0,  0],
#                   [ 0,  0,  0,  0,  0, -1,  2, -1,  0],
#                   [ 0,  0,  0,  0,  0,  0, -1,  2, -1],
#                   [ 0,  0,  0,  0,  0,  0,  0, -1,  2],
#                   ])
# B = -h*np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1]])

#Xi = np.linalg.solve(A, B)
#Psi = np.zeros((Nsample, 1))
Psi = []

def Phi(i, x):
    xnodes = (xgrid[i], xgrid[i+1], xgrid[i+2])
    if xnodes[0] <= x < xnodes[1]:
        return (x - xnodes[0])/h
    if xnodes[1] <= x <= xnodes[2]:
        return -(x - xnodes[2])/h
    if x < xnodes[0] or x > xnodes[2]:
        return 0

def Phi_x(i, x):
    xnodes = (xgrid[i], xgrid[i+1], xgrid[i+2])
    if xnodes[0] <= x < xnodes[1]:
        return 1/h
    if xnodes[1] <= x <= xnodes[2]:
        return -1/h
    if x < xnodes[0] or x > xnodes[2]:
        return 0

# def integration(f, g, x):
#     integral = 0
#     for k in range(0, len(x) - 1):
#         xrand = L*np.random.rand()
#         integral += hsample*f(i, x[k])*g(j, x[k])
#     return integral

A = np.zeros((2*M-1, 2*M-1))
B = np.zeros((2*M-1, 1))

for i in range(0, 2*M-1):
    for j in range(0, 2*M-1):
        for k in range(0, len(xsample) - 1):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            A[i][j] += hsample*Phi_x(i, xsample[k])*Phi_x(j, xsample[k])

for i in range(0, 2*M-1):
    for k in range(0, len(xsample) - 1):
        B[i][0] += -hsample*Phi(i, xsample[k])*xsample[k]
            
            
#print(C)
print(A)
#print(D)
print(B)
val = 0

Xi = np.linalg.solve(A, B)

for j in xsample:
    for i in range(1, 2*M):
        #Psi[j][0] += Xi[i][0]*Phi(i, j) 
        val += Xi[i-1][0]*Phi(i-1, j)
    Psi.append(val)
    val = 0

#print(Psi)

plt.plot(xsample, Psi)
plt.plot(xsample, 1/6*(xsample**3 - xsample), "--")
plt.grid()
plt.savefig("fem.pdf")
plt.show()