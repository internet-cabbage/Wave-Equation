# Wave equation

import numpy as np
import numba
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

Nx, Ny = 500, 200
dh = 0.2
Lx = Nx * dh
Ly = Ny * dh
dt = 0.05

tSteps = 3000
gridShape = (Nx,Ny,tSteps)
c = 1
coEff = (c**2 * dt**2) / (dh**2)

print('CFLCondition means that the coefficient:', coEff, '< 0.5')



def generatorGaussian(sigmaX,sigmaY,muX,muY,t):
    x = np.linspace(-Lx,Lx,Nx)
    y = np.linspace(-Ly,Ly,Ny)

    xGaussian = np.exp(-(x-muX)**2 / (2*sigmaX**2))
    yGaussian = np.exp(-(y-muY)**2 / (2*sigmaY**2))


    initGrid = np.outer(yGaussian,xGaussian)

    waveGrid = np.zeros((Ny,Nx,t), dtype=np.float32)
    waveGrid[:,:,0] = initGrid
    return waveGrid

waveGrid = generatorGaussian(0.6,0.6,0,0,tSteps)



@numba.njit(nogil=True)
def nextState(waveGrid, t):
    ddx = waveGrid[2:,1:-1,t] - 2 * waveGrid[1:-1,1:-1,t] + waveGrid[:-2,1:-1,t]
    ddy = waveGrid[1:-1,2:,t] - 2 * waveGrid[1:-1,1:-1,t] + waveGrid[1:-1,:-2,t]
    waveGrid[1:-1,1:-1,t+1] = 2 * (waveGrid[1:-1,1:-1,t]) - waveGrid[1:-1,1:-1,t-1] + coEff * (ddx + ddy)
    return waveGrid[:,:,t+1]


'''
def exeLoop():
    global waveGrid
    for i in range(0,tSteps-1):
        waveGrid[:,:,i+1] = nextState(waveGrid,i)
exeLoop()
'''
for i in tqdm(range(tSteps-1)):
    waveGrid[:,:,i+1] = nextState(waveGrid,i)

# ===============================================
# Display Code
# ===============================================

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)





# ===============================================
# Animate Code
# ===============================================

def animate(i):
    graph.set_data(waveGrid[:,:,i])
    if i == tSteps-1:
        print('Looped')
    return graph,

minAmp = np.min(waveGrid)
maxAmp = np.max(waveGrid)

print('Min:', minAmp, 'Max:', maxAmp)

graph = ax.imshow(waveGrid[:,:,0], cmap='seismic',vmin=minAmp, vmax=-minAmp,
                  origin='lower')
plt.colorbar(graph,ax=ax)

animation = FuncAnimation(fig, animate, frames=tSteps, blit=True, repeat=True, interval=1)
plt.show()