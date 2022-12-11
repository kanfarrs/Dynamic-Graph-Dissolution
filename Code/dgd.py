"""
Created on Fri Jul 23 11:33:22 2021

@author: kanfar <kanfar@stanford.edu>

"""

# ----External librairies importations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate
import time
import segment
import pyvista as pv
import processDFN
import utilities
import sys
from multiprocess import Pool, Process
from tqdm import tqdm

#import processDFN

class crack:
    """
        CrackDiss is a Python package for the dissolution of a single fracture.
    """
    # class attributes
    n = 4                               # order of non-linear kinetics
    kl = 4*10**-11                      # linear kinetic constant
    kn = 4*10**-8                       # non-linear kinetic constant
    c_s = 1.8*10**-6                    # switch concentration
    c_eq = 2*10**-6                     # equilibrium concentration
    eta = 1.2*10**-2                    # viscosity [g/cms]
    rho = 1                             # densityg/cm^3 [density]
    g = 9.8*100                         # gravitational constant [cm/s^2]
    gamma = 1.7*10**9                   # widening rate constant
    D = 10**-5                          # cm^2/s
    # 145                               # concentration dividor constant (145)
    inc_1 = 100  
    dc = c_s/inc_1                      # step concentration
    dx = 10**-3                         # space grid increment
    init_c = 0

    def __init__(self, num_years, dt):
        self.dt = dt
        self.t = np.arange(0, num_years, dt)
        self.t = np.append(self.t, num_years)
        self.time_condStep = round((len(self.t)-1)/10)

    def calcPerimeter(self, alpha, beta, model):
        # calculates the cross section perimeter across the fracture
        if model == 0:                  # rectangle
            P = 2*alpha + 2*beta
        return P

    def calcR(self, alpha, beta, x, model):
        # calculates flow resistance for the hagen poiseuille equation
        if model == 0:                  # rectangle
            M = 1 - 0.6*alpha/beta
        func = beta*M*(alpha**3)
        R = 12*self.eta/(self.g*self.rho)*np.trapz(1/func, x)
        return R

    def calcFlow(self, R, Hgrad, L):
        # laminar flow
        Q = Hgrad*L/R
        return Q

    def calcDissRate(self, alpha, c):
        # calc diss rate at a specific spatial location in the crack
        # laminar flow
        if c < self.c_s:
            if alpha > 0.1:
                f = (1-c/self.c_eq)*(self.kl /
                                     (1+self.kl*(3*10**-5) /
                                      (3*self.D*self.c_eq)))
            else:
                f = self.kl*(1-c/self.c_eq)
        else:
            f_d = 2*self.D*(self.c_eq-c)/(3*10**-5)
            f_s = self.kn*(1-c/self.c_eq)**self.n
            f = np.amin((f_d, f_s))
        return f

    def updateCrack(self, alpha, beta, f):

        alphaUpdate = alpha + 2*self.gamma*f*self.dt
        betaUpdate = beta + 2*self.gamma*f*self.dt

        return alphaUpdate, betaUpdate

    def loopPlot(self, x, y):
        for i in range(y.shape[1]):
            plt.plot(x, y, '-')

    def plotCrack(self, grid, viewStamps, save = 0):
        # constants
        num_samples = 1000

        # calc
        viewStamps_str = viewStamps.astype(str)
        idx_x = np.linspace(0, len(grid['x'])-1, num_samples).astype(int)
        idx_t = (viewStamps/self.dt).astype(int)
        # plotting properties
        font = {'weight': 'bold',
                'size': 10}
        matplotlib.rc('font', **font)
        # create figure
        fig = plt.figure()
        # plot flow rate
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(grid['t'], grid['Q'], '--b')
        ax1.plot(grid['t'][idx_t], grid['Q'][idx_t], '*r')
        ax1.set_yscale('log')
        ax1.set_title('Flow rate')
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('Q [cm^3/s]')
        # plot aperture
        ax2 = fig.add_subplot(2, 2, 2)
        self.loopPlot(grid['x'][idx_x], grid['alpha'][idx_x, :][:, idx_t])
        ax2.set_yscale('log')
        ax2.set_title('Aperture')
        ax2.set_xlabel('crack length [cm]')
        ax2.set_ylabel('alpha [cm]')
        #ax2.set_ylim(0.001, 0.01)
        #print(min(grid['alpha'][idx_x, :][:, idx_t].all()))
        min_a = grid['alpha'][idx_x, :][:, idx_t].min()
        max_a = grid['alpha'][idx_x, :][:, idx_t].max()
        ax2.set_ylim(min_a, max_a)
        #ax2.set_ylim(0.0001, 1)

        #ax2.set_ylim(np.amin(grid['alpha'][:,0]), np.amax(grid['alpha'][:,-1]))
        ax2.legend(viewStamps_str, prop={'size': 6},
                   title='years', title_fontsize=8)
        # plot concentration ratio
        ax3 = fig.add_subplot(2, 2, 3)
        self.loopPlot(grid['x'][idx_x], grid['c']
                      [idx_x, :][:, idx_t]/self.c_eq)
        ax3.set_title('Concentration ratio')
        ax3.set_xlabel('crack length [cm]')
        ax3.set_ylabel('c / c_eq')
        ax3.legend(viewStamps_str, prop={'size': 6},
                   title='years', title_fontsize=8)
        # plot concentration ratio
        ax4 = fig.add_subplot(2, 2, 4)
        self.loopPlot(grid['x'][idx_x], grid['f']
                      [idx_x, :][:, idx_t]*self.gamma)
        ax4.set_yscale('log')
        ax4.set_title('Widening rate')
        ax4.set_xlabel('crack length [cm]')
        ax4.set_ylabel('wide rate [cm/year]')
        ax4.legend(viewStamps_str, prop={'size': 6},
                   title='years', title_fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 1])  # [0.3, 0.1, 2, 0.95]
        plt.show()

        if save == 1:
            image_name = 'crack_' + (str(grid['x'][-1]) + '_'
                                     + str(self.dt) + '_'
                                     + str(grid['t'][-1])
                                     + '.eps')

            image_format = 'eps'  # e.g .png, .svg, etc.
            fig.savefig(image_name, format=image_format, dpi=1200)

    def cGrid_dc(self, x, init_c):
        """
        vectorizes c  using dc increments given the size of x

        """

        c = np.ones(len(x))*self.dc + init_c
        c[0] = init_c
        c = np.cumsum(c)
        c = np.where(c >= self.c_eq, self.c_eq - self.dc, c)

        return c

    def fGrid_dc(self, c, alpha):
        """
        vectorizes dissolution rate, f, using c and alpha

        """
        f = np.where((alpha > 0.1) & (c < self.c_s),
                     (1-c/self.c_eq)*(self.kl /
                                      (1+self.kl*(3**10**-5) /
                                       (3*self.D*self.c_eq))), self.kl*(1-c/self.c_eq))

        f = np.where((c > self.c_s),
                     np.minimum((2*self.D*(self.c_eq-c)/(3*10**-5)),
                                (self.kn*(1-c/self.c_eq)**self.n)), f)

        return f

    def xGrid_dc(self, f, P, Q, x_ref, L):
        """
        calculates the appropriate spacings dx based on fixed dc 

        """
        i = 0
        x = np.zeros(1)

        while (x[i] < L and i < len(x_ref)):
            #print('test xGrid_dc')
            # find index of the closest value of predefined x (self.x) to x_dx
            # in order to calc ~P at x_dc(iter)
            x_idx = np.argmin(np.abs(x_ref - x[i]))      # closest index
            dx = Q*self.dc/(f[i]*P[x_idx])
            x = np.append(x, x[i] + dx)
            i += 1

        if i == len(x_ref):
            x = x_ref
            #print('approximation failed')

        return x

    def refX_nl(self, x_dc, L):
        """
        non-linear approximation of the linear reference grid x based 
            on the spatial grid calculated in the first iteration 

        """
        resol_factor = 50

        idx_dx = np.arange(len(x_dc))
        idx_x = np.linspace(0, len(x_dc)-1, resol_factor*len(x_dc))

        coef = interpolate.splrep(idx_dx, x_dc)
        x_nl = interpolate.splev(idx_x, coef)

        x_nl = np.delete(x_nl, np.where(x_nl >= L))
        x_nl = np.delete(x_nl, np.where(x_nl < 0))

        x_nl = np.insert(x_nl, 0, 0)
        x_nl = np.append(x_nl, L)

        return x_nl

    #not elegant

    def crackDiss(self, L, init_alpha, init_beta, Hgrad, model=0, algorithm=0):
        """
        calculates crack dissolution using one of three algorithms:
            algorithm = 0: default algorithm. Steps in space is based on
            predefined concentration. 

        """
        if algorithm == 2:
            grid = self.calcDiss_dx(L, init_alpha, init_beta, Hgrad, model)
        else:
            grid = self.calcDiss_dc(
                L, init_alpha, init_beta, Hgrad, model, algorithm)

        return grid

    def createGrid(self, L, init_alpha, init_beta):

        x, alpha, beta = self.crackGrid(L, init_alpha, init_beta)
        Q = np.zeros(len(self.t), dtype='float')
        c = np.ones((len(x), len(self.t)))*self.init_c
        init_f = self.calcDissRate(init_alpha, self.init_c)
        f = np.ones_like(c)*init_f

        grid = {'Q': Q, 'alpha': alpha, 'beta': beta,
                'c': c, 'f': f, 'x': x, 't': self.t}

        return grid

    def crackGrid(self, L, init_alpha, init_beta):

        x = np.arange(0, L, self.dx)
        x = np.append(x, L)  # changed
        # creates crack grid
        alpha = np.ones((len(x), len(self.t) + 1))*init_alpha
        beta = np.ones_like(alpha)*init_beta

        return x, alpha, beta

    def initCrackGrid(self, L, init_alpha, init_beta):

        x = np.arange(0, L, self.dx)
        x = np.append(x, L)  # changed
        # creates spatial dimenstional grid of initial time step
        alpha = np.ones((len(x), 1))*init_alpha
        #alpha = np.ones(len(x))*init_alpha
        beta = np.ones_like(alpha)*init_beta

        return x, alpha, beta

    def approxGrid(self, x, alpha, beta, L, Hgrad, model):
        """
        Approximate spatial grid dimension of all grid properties based on 
            first iteration to decrease computational time

        """
        j = 0
        init_alpha = alpha[0]  # changed
        init_beta = beta[0]  # changed
        Q = np.zeros(len(self.t), dtype='float')

        c_ref = self.cGrid_dc(x, self.init_c)
        f_ref = self.fGrid_dc(c_ref, alpha[:, j])
        P = self.calcPerimeter(alpha[:, j], beta[:, j], model)
        R = self.calcR(alpha[:, j], beta[:, j], x, model)
        Q[j] = self.calcFlow(R, Hgrad, L)

        x_dc = self.xGrid_dc(f_ref, P, Q[j], x, L)
        x_nl = self.refX_nl(x_dc, L)

        alpha_approx = np.ones((len(x_nl), len(self.t) + 1))*init_alpha
        beta_approx = np.ones((len(x_nl), len(self.t) + 1))*init_beta
        c_approx = np.ones((len(x_nl), len(self.t)))*self.init_c
        init_f = self.calcDissRate(init_alpha, self.init_c)
        f_approx = np.ones((len(x_nl), len(self.t)))*init_f

        grid = {'Q': Q, 'alpha': alpha_approx, 'beta': beta_approx,
                'c': c_approx, 'f': f_approx, 'x': x_nl, 't': self.t}

        return grid


    def forward_engine_dx(self, grid, L, R, Hgrad, model, time_step):
        j = time_step

        P = self.calcPerimeter(grid['alpha'][:, j], grid['beta'][:, j], model)
        #R = self.calcR(grid['alpha'][:, j], grid['beta'][:, j], grid['x'], model)
        grid['Q'][j] = self.calcFlow(R, Hgrad, L)
        for i in range(len(grid['x']) - 1):
            dc = self.dx*P[i]*grid['f'][i, j]/grid['Q'][j]
            grid['c'][i+1, j] = grid['c'][i, j] + dc
            if grid['c'][i+1, j] >= self.c_eq:
                grid['c'][i+1, j] = self.c_eq - self.dc
            grid['f'][i+1,
                      j] = self.calcDissRate(grid['alpha'][i+1, j], grid['c'][i+1, j])

        grid['alpha'][:, j+1], grid['beta'][:, j+1] = self.updateCrack(grid['alpha'][:, j],
                                                                       grid['beta'][:, j],
                                                                       grid['f'][:, j])

        return grid

    def forward_engine_dc(self, grid, L, init_c, R, Hgrad, model, time_step):
        j = time_step

        c_ref = self.cGrid_dc(grid['x'], init_c)
        f_ref = self.fGrid_dc(c_ref, grid['alpha'][:, j])

        P = self.calcPerimeter(grid['alpha'][:, j], grid['beta'][:, j], model)
        #R = self.calcR(grid['alpha'][:, j], grid['beta'][:, j], grid['x'], model)
        grid['Q'][j] = self.calcFlow(R, Hgrad, L)

        x_dc = self.xGrid_dc(f_ref, P, grid['Q'][j], grid['x'], L)
        c_dc = self.cGrid_dc(x_dc, init_c)
        grid['c'][:, j] = np.interp(grid['x'], x_dc, c_dc)
        grid['f'][:, j] = self.fGrid_dc(grid['c'][:, j], grid['alpha'][:, j])

        grid['alpha'][:, j+1], grid['beta'][:, j+1] = self.updateCrack(grid['alpha'][:, j],
                                                                       grid['beta'][:, j],
                                                                       grid['f'][:, j])

        return grid

    def calcDiss_dx(self, L, init_alpha, init_beta, Hgrad, model):

        total_start = time.time()
        iter_startTime = time.time()
        time_condition = self.time_condStep
        duration = 0
        grid = self.createGrid(L, init_alpha, init_beta)
        for j in range(len(self.t)):
            R = self.calcR(grid['alpha'][:, j], grid['beta']
                           [:, j], grid['x'], model)
            grid = self.forward_engine_dx(grid, L, R, Hgrad, model, j)

            if j == time_condition:
                iter_endTime = time.time()
                lapse_percent = time_condition/(len(self.t)-1)
                time_condition += self.time_condStep
                duration += iter_endTime - iter_startTime
                print('%.2f percent in %.2f sec' % (lapse_percent,
                                                    duration))
                iter_startTime = time.time()

        total_end = time.time()
        duration = (total_end - total_start)/60
        print('total duration: %.2f min' % duration)

        return grid
    
    def calcDiss_dc(self, L, init_alpha, init_beta, Hgrad, model, algorithm):

        # define grid
        if algorithm == 1:
            grid = self.createGrid(L, init_alpha, init_beta)
        else:  # default or any other choice other than 1 and 2
            x_full, alpha_full, beta_full = self.initCrackGrid(
                L, init_alpha, init_beta)
            grid = self.approxGrid(
                x_full, alpha_full, beta_full, L, Hgrad, model)  # x here is x_

        for j in tqdm(range(len(self.t))):
            R = self.calcR(grid['alpha'][:, j], grid['beta']
                           [:, j], grid['x'], model)
            grid = self.forward_engine_dc(
                grid, L, self.init_c, R, Hgrad, model, j)

        return grid


####################################################################################

class graph:

    def __init__(self, num_years, dt, G, path):
        self.crack = crack(num_years, dt)
        self.seg = utilities.vecTopoints()
        self.t = self.crack.t
        self.init_c = self.crack.init_c
        self.dx = self.crack.dx

        self.G = G
        self.processBoundaryNodes()
        #self.edge_key = list(self.G.edges)
        #self.node_key = list(self.G.nodes)

        # fix
        self.model = 0

        self.source_pres = 1000
        self.target_pres = 0

        #self.dfn = processDFN.dfn()
        self.num_frac = processDFN.dfn(path).getNumPolygons()
        self.domain_size = processDFN.dfn(path).getDomainSize()

        # grid video dimensions
        num_points = 50
        self.num_points = num_points + 1
        # 7.5# 7.5# 8# 9 #9 #actualy year number and not iteration
        self.mapper_cond_start = 30
        self.mapper_iter = 1000  # 4.5# 4.5# 4 #3 #10000

        return

    def graphDiss(self):
        grid = self.calcDiss_dc()
        # self.calcDiss_dc()

        return self.G, grid

    def calcDiss_dc(self):

        mapper_cond = self.mapper_cond_start  # conduct mapper every 5 time steps

        self.graph_list = []
        self.graphGrid()
        # self.graphApproxGrid()
        ####
        self.graphProcessing()
        ####
        for t_step in tqdm(range(len(self.t))):

            ##
            if self.t[t_step] == mapper_cond:
                mapper_cond = mapper_cond + self.mapper_iter
                self.graph_list.append(self.G) #add previous graph
                self.runMapper(t_step)
                print('========= MAPPER RUN =========')
            ###

            self.calcHgrad(t_step)
            self.makeDirected()
            self.getOrderdEdges()

            for edge in self.ordered_edges:
                init_c = self.getInitC(edge, t_step)
                grid = self.edgeTogrid(edge)
                grid = self.crack.forward_engine_dc(grid,
                                                    self.G.edges[edge]['length'],
                                                    init_c,
                                                    1/self.G.edges[edge]['1/R_temp'],
                                                    self.G.edges[edge]['Hgrad_temp'],
                                                    self.model, t_step)
                self.gridToedge(grid, edge)
                
        self.graph_list.append(self.G)  # add last graph if using mapper

        return grid
    

    def runMapperAfter(self):
        
        t_step = len(self.t)
        self.graph_list.append(self.G) #add previous graph
        self.runMapper(t_step)
        self.graph_list.append(self.G)  #add last graph if using mapper
        self.G = self.graph_list[0] #sloppy way to have the graph unchanged in the class after mapper

        return self.graph_list[1]
    
    def runMapper(self, t_step):
        # get the point cloud of current time step
        tic = time.time()
        pcloud = self.parGetGridVox(t_step)
        toc = time.time()
        print('Total PC algorithm done in {:.4f} seconds \n'.format(toc-tic))


        self.QC = pcloud
        # object instance that runs mapper
        tic = time.time()
        mapper = utilities.processMapper(pcloud)
        mapperGraph = mapper.getGraph()
        self.M = self.G.copy() #not necessary I think
        self.G = self.processMapper(mapperGraph)
        toc = time.time()
        print('Mapper algorithm done in {:.4f} seconds \n'.format(toc-tic))

        print('after mapper G nodes are: ', len(self.G.nodes))
        self.mapperGraphGrid()
        print('after mapper G nodes are: ', list(self.G.nodes))
        if nx.is_connected(self.G) == False:
            print('MAPPER GRAPH IS NOT CONNECTED!')
            sys.exit()  # quit()

        return

    def processMapper(self, mapperGraph):
        s_list, t_list = self.findBoundaryNodes()
        print('source nodes are: ', s_list)
        print('target nodes are: ', t_list)
        original_mapper_nodes = list(
            mapperGraph.nodes)  # without boundary nodes

        # careful: before mapper it's a list now it's a numpy array
        # mapper graph does not have boundary nodes. here add the nodes to the graph we save their indicies
        mapperGraph.graph['source_idx'] = np.arange(
            len(mapperGraph.nodes), + len(mapperGraph.nodes) + len(s_list))
        for source_node in s_list:
            mapperGraph = self.addBoundaryNode(
                mapperGraph, source_node, original_mapper_nodes)

        mapperGraph.graph['target_idx'] = np.arange(
            len(mapperGraph.nodes), + len(mapperGraph.nodes) + len(t_list))
        for target_node in t_list:
            mapperGraph = self.addBoundaryNode(
                mapperGraph, target_node, original_mapper_nodes)

        return mapperGraph

    def addBoundaryNode(self, mapperGraph, boundary_node, original_mapper_nodes):
        closest_node, length, boundary_Coords = self.findClosestNode(
            boundary_node, mapperGraph, original_mapper_nodes)

        # this is a new node with no properties
        # change name so that when added to graph it will definitely be added because it's unique. Is there a problem if it's not a number? I don't think so.
        boundary_node = boundary_node + '111' #previously 'b' but had to change cause of karstnet
        mapperGraph.add_node(boundary_node)
        mapperGraph.add_edge(boundary_node, closest_node)

        mapperGraph.edges[(boundary_node, closest_node)]['length'] = length
        mapperGraph.nodes[boundary_node]['alpha'] = mapperGraph.nodes[closest_node]['alpha']
        mapperGraph.nodes[boundary_node]['beta'] = mapperGraph.nodes[closest_node]['beta']
        mapperGraph.nodes[boundary_node]['coords'] = boundary_Coords
        mapperGraph.nodes[boundary_node]['x'] = boundary_Coords[0]
        mapperGraph.nodes[boundary_node]['y'] = boundary_Coords[1]
        mapperGraph.nodes[boundary_node]['z'] = boundary_Coords[2]

        return mapperGraph

    def findBoundaryNodes(self):
        # G is graph before this iteration mapper
        # return: source and target nodes of previous graph (not "s" and "t" but the nodes connected to s and t that are on the boundary)
        node_list = list(self.G.nodes)

        source_idx = self.G.graph['source_idx']
        target_idx = self.G.graph['target_idx']

        source_node_list = [node_list[i] for i in source_idx]
        target_node_list = [node_list[i] for i in target_idx]

        return source_node_list, target_node_list

    def findClosestNode(self, boundary_node, mapperGraph, original_mapper_nodes):
        boundary_coords = self.getCoords(boundary_node)
        #node_list = list(mapperGraph.nodes)
        dist = []
        for node in original_mapper_nodes:
            dist.append(np.linalg.norm(boundary_coords -
                        mapperGraph.nodes[node]['coords']))

        length = min(dist)
        idx_min = dist.index(length)
        closest_node = original_mapper_nodes[idx_min]

        return closest_node, length, boundary_coords

    def getCoords(self, node):
        coords = np.zeros((3,))
        coords[0] = self.G.nodes[node]['x']
        coords[1] = self.G.nodes[node]['y']
        coords[2] = self.G.nodes[node]['z']

        return coords

    def mapperGraphGrid(self):
        self.initNodeAttributes()
        self.initGraphAttributes()
        self.mapperEdgeAttributes()

        return

    def mapperEdgeAttributes(self):
        for edge in list(self.G.edges):
            grid = self.getMapperGrid(edge)
            self.gridToedge(grid, edge)
        return

    def getMapperGrid(self, edge):
        x, alpha, beta = self.getMapperGeometry(edge)
        Q = np.zeros(len(self.t), dtype='float')
        c = np.ones((len(x), len(self.t)))*self.init_c
        init_f = self.crack.calcDissRate(alpha[0, 0], self.init_c)
        #init_f = self.fGrid_dc(c, alpha[:, 0])
        f = np.ones_like(c)*init_f

        grid = {'Q': Q, 'alpha': alpha, 'beta': beta,
                'c': c, 'f': f, 'x': x, 't': self.t}

        return grid

    def getMapperGeometry(self, edge):
        L = self.G.edges[edge]['length']
        x = np.arange(0, L, self.dx)
        x = np.append(x, L)

        node_start = edge[0]
        alpha_start = self.G.nodes[node_start]['alpha']
        beta_start = self.G.nodes[node_start]['beta']

        node_end = edge[1]
        alpha_end = self.G.nodes[node_end]['alpha']
        beta_end = self.G.nodes[node_end]['beta']

        x_at_nodes = [0, L]
        alpha_at_nodes = [alpha_start, alpha_end]
        beta_at_nodes = [beta_start, beta_end]

        # start must always be bigger
        if alpha_start > alpha_end:
            alpha_interp = np.interp(x, x_at_nodes, alpha_at_nodes)
        else:
            alpha_interp = alpha_start

        if beta_start > beta_end:
            beta_interp = np.interp(x, x_at_nodes, beta_at_nodes)
        else:
            beta_interp = beta_start

        #alpha_interp = np.interp(x, x_at_nodes, alpha_at_nodes)
        #beta_interp = np.interp(x, x_at_nodes, beta_at_nodes)

        alpha = np.ones((len(x), len(self.t) + 1))
        alpha = np.transpose(alpha)*alpha_interp
        alpha = np.transpose(alpha)

        beta = np.ones_like(alpha)
        beta = np.transpose(beta)*beta_interp
        beta = np.transpose(beta)

        return x, alpha, beta


    def graphProcessing(self):
        # removes overlapping edges
        # checks if graph is connected after processing

        # collect all edges on each fracture polygon
        self.calcHgrad(0)
        self.makeDirected()
        frac_list = self.getEdgesOnSameFrac()
        redundant_edges = self.getRedundantEdges(frac_list)
        self.removeEdges(redundant_edges)
        self.G = self.G.to_undirected()
        # check if graph is connected
        if nx.is_connected(self.G) == False:
            print('processed graph is not connected')

        return
    
    def removeEdges(self, edges_list):
        for edge in edges_list:
            self.G.remove_edge(edge[0], edge[1])

        return

    def getRedundantEdges(self, frac_list):
        redundant_lst = []
        # loop fractures polygons
        for frac in range(len(frac_list)):
            edge_list = frac_list[frac]
            init_nodes = self.getInitialNodes(edge_list)

            # list of list: y-axis init nodes, x-axis edges starting with init nodes
            sorted_edges_by_nodes = []
            # loop over unique initial nodes of all edges along each fracture
            for node in init_nodes:
                lst = []  # list of edges with similar init_node ID
                # loop edges along each fracture
                for edge in edge_list:
                    if edge[0] == node:
                        lst.append(edge)
                sorted_edges_by_nodes.append(lst)

            redundant_lst.extend(
                self.getRedundantEdgesForEachFrac(sorted_edges_by_nodes))

        return redundant_lst

    def getRedundantEdgesForEachFrac(self, sorted_edges_by_nodes):
        redundant_lst = []
        for edge_list in sorted_edges_by_nodes:
            edge_lengths = []
            for edge in edge_list:
                edge_lengths.append(self.G.edges[edge]['length'])
            min_length = min(edge_lengths)
            idx_min = edge_lengths.index(min_length)
            edge_list.pop(idx_min)
            redundant_lst.extend(edge_list)

        return redundant_lst

    def getInitialNodes(self, edge_list):
        # return unique initial nodes of a given edge list
        init_nodes = []
        for edge in edge_list:
            init_nodes.append(edge[0])
        # remove any duplicates
        init_nodes = set(init_nodes)
        init_nodes = list(init_nodes)

        return init_nodes

    def getEdgesOnSameFrac(self):
        # returns list of lists: each list is a fracture polygon and it contains a list of all edges in that fracture
        # edges here are directed
        frac_list = []
        frac = 1
        while frac <= self.num_frac:
            edge_list = []
            for edge in list(self.G.edges):
                if self.G.edges[edge]['frac'] == frac:
                    edge_list.append(edge)
            if len(edge_list) > 1:  # why
                frac_list.append(edge_list)
            frac = frac + 1
            #print('test get')
        return frac_list

    #######

    def edgeTogrid(self, key):
        grid = {}
        grid['Q'] = self.G.edges[key]['Q']
        grid['c'] = self.G.edges[key]['c']
        grid['f'] = self.G.edges[key]['f']
        grid['alpha'] = self.G.edges[key]['alpha']
        grid['beta'] = self.G.edges[key]['beta']
        grid['x'] = self.G.edges[key]['x']
        grid['t'] = self.t

        return grid

    def gridToedge(self, grid, key):
        # Q, c, f are not predefined and are set here
        self.G.edges[key]['Q'] = grid['Q']
        self.G.edges[key]['c'] = grid['c']
        self.G.edges[key]['f'] = grid['f']
        self.G.edges[key]['alpha'] = grid['alpha']
        self.G.edges[key]['beta'] = grid['beta']
        self.G.edges[key]['x'] = grid['x']

        return

    def graphGrid(self):
        self.initNodeAttributes()
        self.initGraphAttributes()
        self.initEdgeAttributes()

        return

    def graphApproxGrid(self):

        self.initNodeAttributes()
        self.initGraphAttributes()
        self.initEdgeAttributes_t0()

        idx_t = 0
        self.calcHgrad(idx_t)
        self.setApproxGrid()

        return

    def calcHgrad(self, idx_t):

        self.G = self.G.to_undirected()
        self.setRtoEdge(idx_t)
        self.setRtoGraph(idx_t)
        self.setHeadtoNode(idx_t)
        self.setHgradtoEdge(idx_t)
        self.setHgradToGraph(idx_t)
        self.setQtoGraph(idx_t)

        return

    def setApproxGrid(self):
        for key in list(self.G.edges):
            L = self.G.edges[key]['length']
            true_alpha = self.G.edges[key]['alpha']
            true_beta = self.G.edges[key]['beta']
            true_x = self.G.edges[key]['x']
            Hgrad = self.G.edges[key]['Hgrad_temp']
            approx_grid = self.crack.approxGrid(
                true_x, true_alpha, true_beta, L, Hgrad, self.model)
            self.gridToedge(approx_grid, key)

        return

    def initNodeAttributes(self):
        for key in list(self.G.nodes):
            self.G.nodes[key]['head'] = np.zeros(len(self.t))
        return

    def initGraphAttributes(self):
        shape = (self.G.number_of_nodes(),
                 self.G.number_of_nodes(), len(self.t))
        self.G.graph['1/R_adj'] = np.zeros(shape, dtype=float)
        self.G.graph['Hgrad_adj'] = np.zeros(shape, dtype=float)
        self.G.graph['Q_adj'] = np.zeros(shape, dtype=float)

        return

    def initEdgeAttributes(self):
        for key in list(self.G.edges):
            # set up regular grid
            L = self.G.edges[key]['length']
            init_alpha = self.G.edges[key]['alpha']
            init_beta = self.G.edges[key]['beta']
            grid = self.crack.createGrid(L, init_alpha, init_beta)
            self.gridToedge(grid, key)

        return

    def initEdgeAttributes_t0(self):
        for key in list(self.G.edges):
            # set up regular grid
            L = self.G.edges[key]['length']
            init_alpha = self.G.edges[key]['alpha']
            init_beta = self.G.edges[key]['beta']

            (self.G.edges[key]['x'],
             self.G.edges[key]['alpha'],
             self.G.edges[key]['beta']) = self.crack.initCrackGrid(L, init_alpha, init_beta)

        return

    def setRtoEdge(self, t_step):
        # R needs to be computed but not saved (for now I save for QC)
        for key in list(self.G.edges):
            self.G.edges[key]['1/R_temp'] = 1 / self.crack.calcR(self.G.edges[key]['alpha'][:, t_step],
                                                                 self.G.edges[key]['beta'][:, t_step],
                                                                 self.G.edges[key]['x'],
                                                                 self.model)
        return

    def setRtoGraph(self, idx_t):
        # R needs to be computed but not saved (for now I save for QC)
        self.G.graph['1/R_adj'][:, :,
                                idx_t] = nx.to_numpy_array(self.G, weight='1/R_temp')

        return

    def setHeadtoNode(self, idx_t):
        # calculate laplacian
        adjW = self.G.graph['1/R_adj'][:, :, idx_t]
        lapG = self.calcLaplacian(adjW)
        # calculate head
        self.head = self.calcHead(lapG)
        # adds pressure/head attribute to nodes
        # unnecessary step (can directly go to addedgeattribute), but I wanted to save pressures too
        counter = 0
        for key in list(self.G.nodes):
            self.G.nodes[key]['head'][idx_t] = self.head[counter]
            counter += 1

        return

    def calcHead(self, lapG):
        # calculates head

        lapG, vec_zero = self.processInversion(lapG)
        head = np.matmul(np.linalg.inv(lapG), vec_zero)

        self.vec_zero = vec_zero
        self.head = head
        return head

    # change graph to directed
    # think about direction and consequences later
    def setHgradtoEdge(self, idx_t):
        for key in list(self.G.edges):
            start_node = key[0]
            end_node = key[1]
            head1 = self.G.nodes[start_node]['head'][idx_t]
            head2 = self.G.nodes[end_node]['head'][idx_t]
            #Hgrad = head1 - head2
            Hgrad = (head1 - head2)/self.G.edges[key]['length']
            self.G.edges[key]['Hgrad_temp'] = abs(Hgrad)
            # delete this after QC...calculating Q twice (temp and in engine, otherwise when doing 1 step with engine allocate in temp and do 1 redundant step in initializing grid. Either way this in this place needs ot be delted)
            self.G.edges[key]['Q_temp'] = self.crack.calcFlow(1/self.G.edges[key]['1/R_temp'],
                                                              self.G.edges[key]['Hgrad_temp'],
                                                              self.G.edges[key]['length'])
            if head2 < head1:
                self.G.edges[key]['order'] = True
            else:
                self.G.edges[key]['order'] = False
        return

    def setHgradToGraph(self, idx_t):
        self.G.graph['Hgrad_adj'][:, :, idx_t] = nx.to_numpy_array(
            self.G, weight='Hgrad_temp')

        return

    def setQtoGraph(self, idx_t):
        # R needs to be computed but not saved (for now I save for QC)
        self.G.graph['Q_adj'][:, :, idx_t] = nx.to_numpy_array(
            self.G, weight='Q_temp')

        return

    def calcLaplacian(self, adjW):

        self.Rinv = adjW

        degM = np.zeros_like(adjW)  # initialize degree matrix
        degV = np.sum(adjW, axis=1)  # sum over rows to get the degree vector
        # fill degree matrix with degree vector in the diagonal
        np.fill_diagonal(degM, degV)
        lapG = degM - adjW

        self.lapG_unproc = lapG

        return lapG

    def processInversion(self, lapG):
        # assume only 1 inlet and outlet
        # assume DFN saves source then target as the last two nodes
        # assume pressure the same for anything touching the walls

        source_idx = self.G.graph['source_idx']
        target_idx = self.G.graph['target_idx']

        lapG[source_idx, :] = 0
        lapG[target_idx, :] = 0
        lapG[source_idx, source_idx] = 1  # target
        lapG[target_idx, target_idx] = 1  # source

        vec_zero = np.zeros(self.G.number_of_nodes())
        vec_zero[target_idx] = self.target_pres
        vec_zero[source_idx] = self.source_pres

        self.lapG_proc = lapG

        return lapG, vec_zero

    def processBoundaryNodes(self):
        # maybe I can do this in processDFN
        source_neighbors = [neighbor for neighbor in nx.neighbors(self.G, 's')]
        target_neighbors = [neighbor for neighbor in nx.neighbors(self.G, 't')]

        self.G.remove_node('s')
        self.G.remove_node('t')

        node_key = list(self.G.nodes)
        self.G.graph['source_idx'] = [node_key.index(
            neighbor) for neighbor in source_neighbors]
        self.G.graph['target_idx'] = [node_key.index(
            neighbor) for neighbor in target_neighbors]

        return

    def makeDirected(self):
        # directs graph based on pressure differences for concentration accumulation purposes (for flow is unnecesary)
        G_dir = self.G.to_directed()
        # loop based on G which has order information before direction
        for key in list(self.G.edges):
            reversed_key = key[::-1]
            if self.G.edges[key]['order'] == True:
                G_dir.remove_edge(*reversed_key)
            else:
                G_dir.remove_edge(*key)
                G_dir.edges[reversed_key]['order'] = True
        self.G = G_dir

        return

    # if out degree is zero and it's not the target then it's a dead zone and cut all nodes leading there?
    # MAKE MORE simplified and elegant
    # history already taken care of becauase of incoming edges conditions (there will be no repeated visited nodes because nodes won't be processed until all incoming edges are added)
    def getOrderdEdges(self):
        # G is already directed
        # orders edges for correct propagation of concentrations
        node_list = list(self.G.nodes)
        source_idx = self.G.graph['source_idx']
        source_nodes = [node_list[idx] for idx in source_idx]
        ordered_edges = list(self.G.edges(source_nodes))
        edges = ordered_edges.copy()
        fail_nodes = []
        while len(ordered_edges) < len(self.G.edges):
            #print('test get ordered edges')
            # get next nodes
            next_nodes = self.getEndNodeFromEdges(edges)
            if type(fail_nodes) == str:
                print('fail_nodes are:', fail_nodes)
                print('type fail nodes', type(fail_nodes))
                next_nodes.append(fail_nodes)
            else:
                next_nodes.extend(fail_nodes)
            # Processing 1) remove repeated nodes to avoid repeated edges (I think not neccesary edges will be retrieved once for each node)
            next_nodes = list(dict.fromkeys(next_nodes))
            # Processing 2) check for each node all incoming_edges are part of history (ordered_edges), if not, try this node next iteration
            next_nodes, fail_nodes = self.checkOrderHistory(
                ordered_edges, next_nodes)

            if len(next_nodes) == 0:
                next_nodes = fail_nodes  # make a random fail node instead and even
            #print('next nodes', next_nodes)
            #print('fail nodes', fail_nodes)
            edges = list(self.G.edges(next_nodes))
            # save order in ordered_edges
            ordered_edges.extend(edges)

        self.ordered_edges = ordered_edges

        return
    
    def checkOrderHistory(self, ordered_edges, next_nodes):
        # get list of incoming edges to each node
        pass_nodes = []
        fail_nodes = []
        if type(next_nodes) == str:
            print('next_nodes are:', fail_nodes)
            print('type next nodes', type(fail_nodes))
            incoming_edges = self.G.in_edges(next_nodes)
            if set(incoming_edges).issubset(ordered_edges):
                pass_nodes.append(next_nodes)
            else:
                fail_nodes.append(next_nodes)
        else:
            for node in next_nodes:
                incoming_edges = self.G.in_edges(node)
                if set(incoming_edges).issubset(ordered_edges):
                    pass_nodes.append(node)
                else:
                    fail_nodes.append(node)

        return pass_nodes, fail_nodes

    def getEndNodeFromEdges(self, edges_list):
        end_nodes = []
        for edge in edges_list:
            end_nodes.append(edge[1])

        return end_nodes

    def getInitC(self, edge, idx_t):

        # find junction node and edges going into and outside of node
        start_node = edge[0]
        in_edges = list(self.G.in_edges(start_node))
        out_edges = list(self.G.out_edges(start_node))

        # calculate total concentration into node
        c_list = [self.G.edges[key]['c'][-1, idx_t] for key in in_edges]
        total_c = sum(c_list)

        # calculate fraction of concentration going into edge
        # can be flow sum of in edges instead and then no need to calc out_edges in this func
        flow_list = [self.G.edges[key]['Q_temp'] for key in out_edges]
        total_flow = sum(flow_list)
        edge_flow = self.G.edges[edge]['Q_temp']
        frac = edge_flow/total_flow

        init_c = frac*total_c

        return init_c

    def graphPlotCrack(self, num_edges, viewStamps, save=0):
        # limit plot to first 3 edges for now
        edge_list = list(self.ordered_edges)
        for i in range(num_edges):
            edge = edge_list[i]
            grid = self.edgeTogrid(edge)
            self.crack.plotCrack(grid, viewStamps, save)
        return

    def QCplotInitC(self, viewStamps):
        plt.figure()
        # find idx of time
        flag = np.in1d(self.t, viewStamps)
        t_idx_array = np.argwhere(flag)
        # label of edge for plotting purposes
        edge_label = np.arange(len(self.ordered_edges))
        # label of time for plotting purposes
        viewStamps_str = viewStamps.astype(str)
        for t_step in range(len(t_idx_array)):
            # number of init_c same as number of edges
            init_c = np.zeros(len(self.ordered_edges))
            idx = 0
            for edge in self.ordered_edges:  # change over time, need to save or just plot random edges
                # init_c is the first concentration in the edge [0]
                init_c[idx] = self.G.edges[edge]['c'][0, t_step]
                idx = idx + 1
            plt.plot(edge_label, init_c, '-o')
            plt.xlabel('edge # based on ordered edges')
            plt.ylabel('initial concentration')
            plt.legend(viewStamps_str)

        return

    def gridVideoVanilla(self, filename="video_test.mp4"):
        time_list = self.getGeometryCloud()

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        dim = [self.num_points, 2, 2]  # why does this work?

        for i in range(len(self.t)):
            plotter.clear()
            edge_list = time_list[i]
            dic = {}
            for j in range((len(self.G.edges))):  # make len(edge_list). it's the same
                edge_cloud = edge_list[j]

                dic["%s" % j] = pv.StructuredGrid()
                dic["%s" % j].points = edge_cloud
                dic["%s" % j].dimensions = dim
                #voxel = pv.voxelize(dic["%s" %j], check_surface=False)
                #plotter.add_mesh(voxel, show_edges = True)

                plotter.add_mesh(dic["%s" % j], show_edges=True)

            plotter.show_grid()
            plotter.add_text(f"Time: {self.t[i]} years", name='time-label')
            plotter.write_frame()  # write initial data

        # plotter.close()

        return

    def getGridVox(self, t_step):
        density_factor = 25  # 20
        dim = [self.num_points, 2, 2]  # why does this work?

        edge_list = self.getGeometryCloudAtTime(t_step)

        # process first then proceed with rest starting from 1
        edge_cloud = edge_list[0]
        mesh = pv.StructuredGrid()
        mesh.points = edge_cloud
        mesh.dimensions = dim
        mesh = mesh.extract_surface()
        mesh = mesh.smooth()
        vox = pv.voxelize(mesh, check_surface=False, density=mesh.length/(mesh.length*density_factor))
        
        for j in range(1, (len(self.G.edges))):  # make len(edge_list). it's the same
            print(j)
            edge_cloud = edge_list[j]
            mesh = pv.StructuredGrid()
            mesh.points = edge_cloud
            mesh.dimensions = dim
            mesh = mesh.extract_surface()
            #mesh = mesh.smooth()
            vox = vox + pv.voxelize(mesh, check_surface=False,
                                    density=mesh.length/(mesh.length*density_factor))

        return vox.points
    

    def edgeToVox(self, edge_cloud):
        # each func returns vox.points
        # then st
        # can I stack the voxes af
    
        dim = [self.num_points, 2, 2]  # why does this work?
        density_factor = 1  # 20

        mesh = pv.StructuredGrid()
        mesh.points = edge_cloud
        mesh.dimensions = dim
        mesh = mesh.extract_surface()
        #mesh = mesh.smooth()
        vox = pv.voxelize(mesh, check_surface=False, density=mesh.length/(mesh.length*density_factor))
        pc = vox.points
        #can decimate if too expensive 
        #pc = utilities.dataProcess().decimatePoints(pc, 500)
    
        return pc
    
    def decimatePoints(self, points, num_points): #duplicate function in mapper
        #should put this in utilities
        all_idx_random = np.random.permutation(len(points))
        idx_keep = all_idx_random[:num_points]
        points = points[idx_keep]
    
        return points

    def parGetGridVox(self, t_step):
        tic = time.time()
        edge_list = self.getGeometryCloudAtTime(t_step)
        toc = time.time()
        print('getGeometryCloudAtTime Done in {:.4f} seconds \n'.format(toc-tic))
        
        tic = time.time()
        p = Pool()
        lst_pc = p.map(self.edgeToVox, edge_list)  # for single input you can do [n]
        pc = np.vstack(lst_pc)
        toc = time.time()
        print('parEdgeToVox Done in {:.4f} seconds \n'.format(toc-tic))

        tic = time.time()
        pc = utilities.dataProcess().pcUniformDensityXY(pc)
        toc = time.time()
        print('pcUniformDensityXY Done in {:.4f} seconds \n'.format(toc-tic))
        
        return pc 

    def gridVideoVox(self, filename="video_test.mp4", density_factor=20):
        # maybe the density factor should be the same across all? make constant?
        # how to plot grid on
        time_list = self.getGeometryCloud()

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        dim = [self.num_points, 2, 2]  # why does this work?

        for i in range(len(self.t)):
            # plotter.clear()
            edge_list = time_list[i]
            # process first then proceed with rest starting from 1
            edge_cloud = edge_list[0]
            mesh = pv.StructuredGrid()
            mesh.points = edge_cloud
            mesh.dimensions = dim
            mesh = mesh.extract_surface()
            mesh = mesh.smooth()
            vox = pv.voxelize(mesh, check_surface=False,
                              density=mesh.length/(mesh.length*density_factor))
            for j in range(1, (len(self.G.edges))):  # make len(edge_list). it's the same
                edge_cloud = edge_list[j]
                mesh = pv.StructuredGrid()
                mesh.points = edge_cloud
                mesh.dimensions = dim
                mesh = mesh.extract_surface()
                mesh = mesh.smooth()
                vox = vox + pv.voxelize(mesh, check_surface=False,
                                        density=mesh.length/(mesh.length*density_factor))

            plotter.add_mesh(vox, color=True, show_edges=True, opacity=1)
            plotter.reset_camera()
            plotter.show_grid()
            plotter.add_text(
                f"Time: {np.round(self.t[i],1)} years", name='time-label')
            plotter.write_frame()  # write initial data

        plotter.close()

        return

    def gridVideo(self, filename="video_test.mp4"):
        # can add smooth if you want
        # how to plot grid on
        self.processInletGeometry()
        time_list = self.getGeometryCloud()

        # step = 10 #number of time idx to add to 0 each iteration
        #time_list = self.getGeometryCloud(step)

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        dim = [self.num_points, 2, 2]  # why does this work?

        # for i in range(0, len(self.t), step):
        for i in tqdm(range(len(self.t))):
            surf = pv.StructuredGrid()
            surf = surf.extract_surface()

            edge_list = time_list[i]
            for j in range((len(self.G.edges))):  # make len(edge_list). it's the same
                edge_cloud = edge_list[j]
                mesh = pv.StructuredGrid()
                mesh.points = edge_cloud
                mesh.dimensions = dim
                temp = mesh.extract_surface()
                surf = surf + temp

            surf = surf.smooth()
            #surf = surf.smooth(n_iter = 15)
            plotter.add_mesh(surf)  # , show_edges = True)
            plotter.show_grid()
            plotter.add_text(
                f"Time: {np.round(self.t[i],1)} years", name='time-label')
            plotter.write_frame()  # write initial data

        plotter.close()

        return
    
    def getEdgeSurf(self, edge_cloud):
        
        surf = pv.StructuredGrid()
        surf = surf.extract_surface()
        dim = [self.num_points, 2, 2]  # why does this work?
        mesh = pv.StructuredGrid()
        mesh.points = edge_cloud
        mesh.dimensions = dim
        surf = mesh.extract_surface()
        
        return surf
    
    def getTotalSurf(self, surf_lst):
        
        surf = pv.StructuredGrid()
        surf = surf.extract_surface()
        
        for edge_surf in surf_lst:
            surf = surf + edge_surf
            
        
        return surf
    
    def parGridVideo(self, filename="video_test.mp4"):
        
        #return list of surfaces for each edge
        #add the surfaces into one object
        #loop and write frame for each time sequentially
        
        self.processInletGeometry()
        time_list = self.parGetGeometryCloud()

        # step = 10 #number of time idx to add to 0 each iteration
        #time_list = self.getGeometryCloud(step)

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        
        for i in range(len(self.t)):
            edge_list = time_list[i]
            p = Pool()
            surf_lst = p.map(self.getEdgeSurf, edge_list)  # for single input you can do [n]
            surf = self.getTotalSurf(surf_lst)
    
            surf = surf.smooth()
            plotter.add_mesh(surf)  # , show_edges = True)
            plotter.show_grid()
            plotter.add_text(
                f"Time: {np.round(self.t[i],1)} years", name='time-label')
            plotter.write_frame()  # write initial data
            #tic = time.time()
            #toc = time.time()
            #print('point cloud retrieved in {:.4f} seconds \n'.format(toc-tic))
        return

    def gridVideoMapper(self, filename="video_test.mp4"):
        # can add smooth if you want
        # how to plot grid on

        #step = 1

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        dim = [self.num_points, 2, 2]  # why does this work?

        # for i in range(0, len(self.t), step*self.crack.dt):

        actor = dict()
        graph_counter = 0
        mapper_cond = self.mapper_cond_start
        self.G = self.graph_list[graph_counter]
        self.processInletGeometry()
        time_list = self.getGeometryCloud()

        for i in range(len(self.t)):

            if self.t[i] == mapper_cond:
                mapper_cond = mapper_cond + self.mapper_iter
                graph_counter += 1
                print('graph_counter', graph_counter)
                self.G = self.graph_list[graph_counter]
                self.processInletGeometry()
                time_list = self.getGeometryCloud()

            surf = pv.StructuredGrid()
            surf = surf.extract_surface()

            edge_list = time_list[i]
            for j in range((len(self.G.edges))):  # make len(edge_list). it's the same
                edge_cloud = edge_list[j]
                mesh = pv.StructuredGrid()
                mesh.points = edge_cloud
                mesh.dimensions = dim
                temp = mesh.extract_surface()
                surf = surf + temp

            #surf = surf.smooth()
            surf = surf.smooth(n_iter=15)
            actor[i] = plotter.add_mesh(surf)  # , show_edges = True)
            if i > 1:
                for j in range(i):
                    plotter.remove_actor(actor[j])

           # plotter.add_mesh(surf)
            plotter.show_grid()
            plotter.add_text(
                f"Time: {np.round(self.t[i],1)} years", name='time-label')
            plotter.write_frame()  # write initial data
            del surf
            # if self.t[i] + 0.1 == 12 or self.t[i] == 12:
            #     plotter.save_graphic("img.eps")

        plotter.close()

        return
    
    def gridMapperEndSnap(self, filename="video_test.mp4"):

        self.processInletGeometry()
        edge_list = self.getGeometryCloudAtTime(len(self.t)-1)

        plotter = pv.Plotter()
        plotter.open_movie(filename)
        dim = [self.num_points, 2, 2]  # why does this work?

        surf = pv.StructuredGrid()
        surf = surf.extract_surface()

        for j in range((len(self.G.edges))):  # make len(edge_list). it's the same
            edge_cloud = edge_list[j]
            mesh = pv.StructuredGrid()
            mesh.points = edge_cloud
            mesh.dimensions = dim
            temp = mesh.extract_surface()
            surf = surf + temp
        surf = surf.smooth()
        #surf = surf.smooth(n_iter = 15)
        plotter.add_mesh(surf)  # , show_edges = True)
        plotter.show_grid()
        plotter.add_text(
            f"Time: {np.round(self.t[-1],1)} years", name='time-label')
        plotter.write_frame()  # write initial data

        #plotter.close()
        
        return

    def getGeometryCloudAtTime(self, t_step):

        space_lst = []
        for edge in list(self.G.edges):  # self.ordered_edges):
            points = self.getEdgeGeometry(edge, t_step)
            space_lst.append(points)

        return space_lst

    def getGeometryCloud(self):  # ,step
        # return time_lst: a list of lists.
        # 1st list is time.
        # 2nd list are edges.
        # Inside the edge list are the geometry points of the edges.
        # to do
        # instead of append we can do pre-allocate time and space
        # getEdgeGeometry before time step and then use the time step to just get the 4 corners from alpha and beta (point x-spacings doesn't change over time)
        time_lst = []
        for t_step in range(len(self.t)):
            print(t_step)
            # for t_step in range(0, len(self.t), step):
            space_lst = []
            # list(self.ordered_edges): #should be directed since after last iteration
            for edge in list(self.G.edges):
                points = self.getEdgeGeometry(edge, t_step)
                ###
                #points = self.addNoise(points)
                ###
                space_lst.append(points)
            time_lst.append(space_lst)
        return time_lst
    
    def parGetGeometryCloud(self):  # ,step
        # return time_lst: a list of lists.
        # 1st list is time.
        # 2nd list are edges.
        # Inside the edge list are the geometry points of the edges.
        # to do
        # instead of append we can do pre-allocate time and space
        # getEdgeGeometry before time step and then use the time step to just get the 4 corners from alpha and beta (point x-spacings doesn't change over time)
        time_lst = []
        edge_lst = list(self.G.edges)
        for t_step in range(len(self.t)):   
            self.pool_t = t_step
            p = Pool()
            space_lst = p.map(self.getEdgePoints, edge_lst)  # for single input you can do [n]
            time_lst.append(space_lst)
        return time_lst
    
    def getEdgePoints(self, edge):
        #double input pool
        
        points = self.getEdgeGeometry(edge, self.pool_t)
        
        return points
    
    

    def addNoiseEachPixel(self, points):
        factor = 0.01
        points[:, 0] = points[:, 0] + \
            np.random.normal(0, np.abs(factor*points[:, 0]))
        points[:, 1] = points[:, 1] + \
            np.random.normal(0, np.abs(factor*points[:, 1]))
        points[:, 2] = points[:, 2] + \
            np.random.normal(0, np.abs(factor*points[:, 2]))

        return points

    # def addNoise(self, points, factor = 0.1 , num_points = 50):
    #     num_iter = int(np.floor(len(points)/num_points))
    #     for i in range(num_iter):
    #     #if num points is number then it will cause problem if edge is too small just like processinlet geoemetry
    #         points[i*num_points:(i+1)*num_points,0] = points[i*num_points:(i+1)*num_points,0] + np.random.normal(0, np.abs(factor*points[i*num_points:(i+1)*num_points, 0]))
    #         points[i*num_points:(i+1)*num_points,1] = points[i*num_points:(i+1)*num_points,1] + np.random.normal(0, np.abs(factor*points[i*num_points:(i+1)*num_points, 1]))
    #         points[i*num_points:(i+1)*num_points,2] = points[i*num_points:(i+1)*num_points,2] + np.random.normal(0, np.abs(factor*points[i*num_points:(i+1)*num_points, 2]))

    #     return points

    def addNoise(self, points, factor=0.01, num_points=10):
        num_iter = int(np.floor(len(points)/num_points))
        for i in range(num_iter):
            # if num points is number then it will cause problem if edge is too small just like processinlet geoemetry
            points[i*num_points:(i+1)*num_points, 0] = points[i*num_points:(
                i+1)*num_points, 0] + np.random.normal(0, np.abs(factor*points[i*num_points, 0]))
            points[i*num_points:(i+1)*num_points, 1] = points[i*num_points:(
                i+1)*num_points, 1] + np.random.normal(0, np.abs(factor*points[i*num_points, 1]))
            points[i*num_points:(i+1)*num_points, 2] = points[i*num_points:(
                i+1)*num_points, 2] + np.random.normal(0, np.abs(factor*points[i*num_points, 2]))

        return points

    def getEdgeGeometry(self, edge, t_step):
        # if inclination is 90 or 0 we do corners mannual
        # otherwise we do this
        # test with different orientation fracture dissolution

        # get edge points (directed based on ordered edges)
        # azimuth is between 0-180 from x, inclination is 0-90 from z (always positive angles)
        # does not need to be done every time step
        points, azi, inc = self.getPointsAlongEdge(edge)
        # limit points to certain indices
        idx = self.getIndex(edge)
        inc = np.pi/2 - inc  # more intuitive from zero

        # later implement more accurate geometry using azimuth and inclination
        if azi < np.radians(60):
            if inc > np.radians(60):  # vertical
                # right top
                p1 = points[idx, :]
                p1[:, 0] = p1[:, 0] + self.G.edges[edge]['beta'][idx, t_step]
                p1[:, 1] = p1[:, 1] + self.G.edges[edge]['alpha'][idx, t_step]
                # left top
                p2 = points[idx, :]
                p2[:, 0] = p2[:, 0] + self.G.edges[edge]['beta'][idx, t_step]
                p2[:, 1] = p2[:, 1] - self.G.edges[edge]['alpha'][idx, t_step]
                # right down
                p3 = points[idx, :]
                p3[:, 0] = p3[:, 0] - self.G.edges[edge]['beta'][idx, t_step]
                p3[:, 1] = p3[:, 1] + self.G.edges[edge]['alpha'][idx, t_step]
                # left down
                p4 = points[idx, :]
                p4[:, 0] = p4[:, 0] - self.G.edges[edge]['beta'][idx, t_step]
                p4[:, 1] = p4[:, 1] - self.G.edges[edge]['alpha'][idx, t_step]
            else:
                # right top
                p1 = points[idx, :]
                p1[:, 2] = p1[:, 2] + self.G.edges[edge]['beta'][idx, t_step]
                p1[:, 1] = p1[:, 1] + self.G.edges[edge]['alpha'][idx, t_step]
                # left topå
                p2 = points[idx, :]
                p2[:, 2] = p2[:, 2] + self.G.edges[edge]['beta'][idx, t_step]
                p2[:, 1] = p2[:, 1] - self.G.edges[edge]['alpha'][idx, t_step]
                # right down
                p3 = points[idx, :]
                p3[:, 2] = p3[:, 2] - self.G.edges[edge]['beta'][idx, t_step]
                p3[:, 1] = p3[:, 1] + self.G.edges[edge]['alpha'][idx, t_step]
                # left down
                p4 = points[idx, :]
                p4[:, 2] = p4[:, 2] - self.G.edges[edge]['beta'][idx, t_step]
                p4[:, 1] = p4[:, 1] - self.G.edges[edge]['alpha'][idx, t_step]
        else:
            if inc > np.radians(60):  # vertical
                # right top
                p1 = points[idx, :]
                p1[:, 1] = p1[:, 1] + self.G.edges[edge]['beta'][idx, t_step]
                p1[:, 0] = p1[:, 0] + self.G.edges[edge]['alpha'][idx, t_step]
                # left top
                p2 = points[idx, :]
                p2[:, 1] = p2[:, 1] + self.G.edges[edge]['beta'][idx, t_step]
                p2[:, 0] = p2[:, 0] - self.G.edges[edge]['alpha'][idx, t_step]
                # right down
                p3 = points[idx, :]
                p3[:, 1] = p3[:, 1] - self.G.edges[edge]['beta'][idx, t_step]
                p3[:, 0] = p3[:, 0] + self.G.edges[edge]['alpha'][idx, t_step]
                # left down
                p4 = points[idx, :]
                p4[:, 1] = p4[:, 1] - self.G.edges[edge]['beta'][idx, t_step]
                p4[:, 0] = p4[:, 0] - self.G.edges[edge]['alpha'][idx, t_step]
            else:
                # right top
                p1 = points[idx, :]
                p1[:, 2] = p1[:, 2] + self.G.edges[edge]['alpha'][idx, t_step]
                p1[:, 0] = p1[:, 0] + self.G.edges[edge]['beta'][idx, t_step]
                # left top
                p2 = points[idx, :]
                p2[:, 2] = p2[:, 2] + self.G.edges[edge]['alpha'][idx, t_step]
                p2[:, 0] = p2[:, 0] - self.G.edges[edge]['beta'][idx, t_step]
                # right down
                p3 = points[idx, :]
                p3[:, 2] = p3[:, 2] - self.G.edges[edge]['alpha'][idx, t_step]
                p3[:, 0] = p3[:, 0] + self.G.edges[edge]['beta'][idx, t_step]
                # left down
                p4 = points[idx, :]
                p4[:, 2] = p4[:, 2] - self.G.edges[edge]['alpha'][idx, t_step]
                p4[:, 0] = p4[:, 0] - self.G.edges[edge]['beta'][idx, t_step]

        all_points = np.concatenate((p1, p2, p3, p4), axis=0)

        return all_points

    ##########

    def processInletGeometry(self):
        num_points = 1  # points for inlet processing
        self.H = self.G.copy()  # save G before we play with it
        edge_list = list(self.H.edges)
        # can be parallelized
        for idx_t in range(len(self.t)):
            for edge in edge_list:
                self.G.edges[edge]['alpha'][0:num_points,
                                            idx_t] = self.G.edges[edge]['alpha'][num_points, idx_t]
                self.G.edges[edge]['beta'][0:num_points,
                                           idx_t] = self.G.edges[edge]['beta'][num_points, idx_t]


    def boundGeometry(self, points):
        # constrain geometry to domain size
        for coord_idx in range(3):
            points[:, coord_idx] = np.where(points[:, coord_idx] > 0.5*self.domain_size[coord_idx],
                                            0.5*self.domain_size[coord_idx],
                                            points[:, coord_idx])
            points[:, coord_idx] = np.where(points[:, coord_idx] < -0.5*self.domain_size[coord_idx],
                                            -0.5*self.domain_size[coord_idx],
                                            points[:, coord_idx])

        return points

    def getIndex(self, edge):
        L = self.G.edges[edge]['length']
        x = self.G.edges[edge]['x']
        #num_points = 50
        num_points = self.num_points - 1

        steps = L/num_points
        steps_acum = np.arange(num_points)*steps
        steps_acum = np.append(steps_acum, L)

        x_matrix = np.transpose([x] * (num_points + 1))
        idx = np.argmin((np.abs(x_matrix - steps_acum)), axis=0)

        # self.num_points = num_points + 1 #doesn't change based on edge

        return idx

    def getPointsAlongEdge(self, edge):

        start_node = edge[0]
        start_node_coords = self.getNodeCoords(start_node)

        end_node = edge[1]
        end_node_coords = self.getNodeCoords(end_node)

        edge_points, azi, inc = self.seg.irregular(
            start_node_coords, end_node_coords, self.G.edges[edge]['x'])

        return edge_points, azi, inc

    def getNodeCoords(self, node):
        x = self.G.nodes[node]['x']
        y = self.G.nodes[node]['y']
        z = self.G.nodes[node]['z']
        coords = np.array([x, y, z])

        return coords
