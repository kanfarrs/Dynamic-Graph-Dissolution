from dgd import crack, graph
import numpy as np 
from processDFN import dfn
import os

def crackDiss():
    #crackDiss params
    num_years = 20000               #[years]
    dt = 10                         #[years]
    init_alpha = 0.02               #[cm]
    init_beta = 100                 #[cm]
    length = 1e4                    #[cm] if larger, may have to load spatial array instead of initializing it
    hgrad = 0.01                    #[unitless]
    
    #run crackDiss
    diss = crack(num_years, dt)
    grid = diss.crackDiss(length, init_alpha, init_beta, hgrad)

    #plot results
    viewStamps = np.array([0, 10000, 17000, 17250, 17500])   #[years]
    diss.plotCrack(grid, viewStamps)
    
    return grid

def graphDiss():
    path = os.getcwd() + '/data/'
    G = dfn(path).getGraph()
    
    num_years = 10
    dt = 1
    
    diss_graph = graph(num_years, dt, G, path).graphDiss()
    
    return diss_graph
    
if __name__ == "__main__":
    #crackDiss()
    graphDiss()

