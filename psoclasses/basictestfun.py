import numpy as np
def expsins(X):
    x=X[:,0]
    y=X[:,1]
    z=np.sin(np.sqrt(x**2+y**2))/(np.sqrt(x**2+y**2))+np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)-2.71289
    return -1*z#max形式
def rast(X):#Rastrigin
    A=10
    x=X[:,0]
    y=X[:,1]
    z=2*A+x**2-A*np.cos(2*np.pi*x)+y**2-A*np.cos(2*np.pi*y)
    return z
def rasthidim(X):#Rastrigin high dim
    A=10
    z=0
    for i in range(X.shape[1]):
        z=z+X[:,i]**2-A*np.cos(2*np.pi*X[:,i])+10
    return z
