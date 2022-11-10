import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

args = {'R1':1.3,'R2':3.44,'R3':0.043,'R4':0.193,'alphaf':11.6,'alphar':11.6,'c':4.4,'I':22.5*10**(-6)}#R1,R2,R3,R4,alphaf,alphar,c,Ir

def Derivative(Params,t,args=args):
    VM1,VM2,IM = Params[:3] #Unpacking the Master parameters
    VS1,VS2,IS = Params[3:6] #Unpacking the Slave Parameters
    GM = G(VM1-VM2)
    GS = G(VS1-VS2) 
    #print(GM,GS)
    return VM1/args['R1'] - GM, GM - IM, VM2 - args['R4']*IM, VS1/args['R1'] - GS, GS - IS + args['c']*(VM2-VS2), VS2 - args['R4']*IS #Derivatives

def G(V,args=args):
    return V/args['R2']+args['I']*(np.exp(args['alphaf']*V)-np.exp(-args['alphar']*V))


t = 23.5*10**(-6)*np.linspace(0,1e8,int(1e5)) #Setting the time
# Params = np.array([1.2,1. ,0.3*10**(-3),1.,1.,0.3*10**(-3)]) #setting the parameters, Note: It may be important to play with the initial conditions a bit
Params = np.array([1,1, 0.3 * 10**(-3),1, 1, 0.3 * 10**(-3)])
states = odeint(Derivative,Params,t)   #Solving the odeint
Xpar = (states[:,:3]+states[:,3:6])/2  #Perpendicular Projections for our 3d space
Xperp = (states[:,:3]-states[:,3:6])/2 #Parallel Projections for our 3d space

fig  = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')

num = int(1e4)
ax.set(xlabel='XPERP1', ylabel='XPAR1', zlabel='XPAR3')
ax.scatter3D(Xperp[:,0][:num], Xpar[:,0][:num], Xpar[:,2][:num])

plt.show()

'''
fig = plt.figure()
ax  = plt.axes(projection='3d')

def animate_func(n):
    # clear figure to update line
    ax.clear()
    
    # plot trajectory
    ax.plot3D(Xperp[:, 0][:n+1], Xpar[:,0][:n+1], Xpar[:,1][:n+1], c='blue')
    
    # update plot point location
    ax.scatter(Xperp[:,0][n], Xpar[:,0][n], Xpar[:,1][n], c='black', marker='o')
    
    # add constant origin
    ax.plot3D(Xperp[:,0][0], Xpar[:,0][0], Xpar[:,1][0], c='red', marker='o')
    
    # set axes limits
    ax.set_xlim3d([-0.08, 0.01])
    ax.set_ylim3d([-2., 2.])
    ax.set_zlim3d([-1., 1.])
    
    # add figure labels
    ax.set_title('Dragon King Trajectory \nTime= ' + str(np.round(t[n], decimals=2)) + ' sec')
    
    ax.set_xlabel('Xperp')
    ax.set_ylabel('Xpar1')
    ax.set_zlabel('Xpar2')
    
    return fig,

# interval: delay between frames in milliseconds
ani = FuncAnimation(fig, animate_func, interval=10, frames=len(t))
plt.show()'''
