#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:11:20 2022

@author: kanfar
"""

import kmapper as km
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import open3d as o3d
import sklearn
import karstnet as kn
from kneebow.rotor import Rotor
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import utilities
import seaborn as sns
import scipy
import math
                

class processMapper:
    
    num_points = 12000 #10000
    n_cubes = 18
    overlap = 0.5 #0.5
    std_ratio = 1 
    fraction = 0.01              #fraction of num_points to calculate average distance from points
        
    def __init__(self, data):
        #graph: dict graph returned by mapper
        #data: processed point cloud over which mapper is computed
        self.data = self.processData(data)
        self.graph = self.runMapper()           #graph is dictionary from mapper not networkx graph
        self.postMapperProcessing()
        
    def processData(self, data):

        #std_ratio low:
            #just keep points that have average distances close to mean
            #in this case, std can very low cause most point around mean (the distribution is very narrow around the mean)
        #nb_neighbors:
            #number of neighbors to which to calculate average distance at this point
            #outliers will be more sensitive to more points because the distance will be high
        
        #1) decimate data
        #data = self.decimatePoints(data, self.num_points)
        data = utilities.dataProcess().decimatePoints(data, self.num_points)

        #2) remove noisy voxels
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(data)
        cl, ind = pc.remove_statistical_outlier(nb_neighbors=int(self.num_points*self.fraction), std_ratio=self.std_ratio)
        
        return data[ind]
        
    def runMapper(self):
        
        #default clustering is density based
        
        # Initialize
        self.mapper = km.KeplerMapper(verbose=2) #you can also just inheret the class    
        # Fit to and transform the data
        projected_data = self.mapper.fit_transform(self.data, projection=[0,1,2]) # X-Y axis
        # Create a cover with 10 elements
        cover = km.Cover(n_cubes = [self.n_cubes,1,1], perc_overlap = [self.overlap, self.overlap, self.overlap])
        # Create dictionary called 'graph' with nodes, edges and meta-information
        
        #eps = utilities.dataProcess().getDBSCANeps(self.data) 
        graph = self.mapper.map(projected_data, self.data, cover=cover, remove_duplicate_nodes=True, 
                        #clusterer = sklearn.cluster.KMeans(n_clusters=3))
                         clusterer = sklearn.cluster.DBSCAN(eps=1.8)) 
        #if data is mentioned then supercedes projection
        
        
        # graph = self.mapper.map(self.data, cover=cover, remove_duplicate_nodes=True, 
        #                         clusterer = sklearn.cluster.DBSCAN(eps=0.25))
        # graph = self.mapper.map(projected_data, cover=cover, remove_duplicate_nodes=True, 
        #                         clusterer = sklearn.cluster.DBSCAN(eps=0.25)) #0.25 default eps is 0.5, previously I didn't input clusterer

        return graph
    
    def postMapperProcessing(self):
        #change name of the graph
        self.G = km.to_networkx(self.graph)
        self.getNodeInfo()
        self.getLength()
        self.removeIsolatedNodes()
        self.G = nx.convert_node_labels_to_integers(self.G, 0)
        self.changeIDtoString() #not necessary
        self.smoothGeometry()
        self.delDeg1() #rational is that if it's actually not an artifact from clustering then there will still be 1 degree for that feature even after removing 

        return 
    
    def delDeg1(self):
        #does it actually solve the problem?
        remove = [node for node,degree in dict(self.G.degree()).items() if degree == 1]
        self.G.remove_nodes_from(remove)
        
        return
        
    
    def smoothGeometry(self):
        #generated pointcloud can have outlier points that make the geometry larger than what it is
        #this function average every node by the closest two other nodes it's connected to
        for node in list(self.G.nodes):
            #secondary neighbor option
            # neighbors = list(self.G.neighbors(node))
            # secondary_neighbors = []
            # for neighbor in neighbors:
            #     secondary_neighbors.extend(list(self.G.neighbors(neighbor)))
            # neighbors.extend(secondary_neighbors)
            # print(neighbors)
            # alpha = []
            # beta = []
            # #alpha.append(self.G.nodes[node]['alpha'])
            # #beta.append(self.G.nodes[node]['beta'])
            # for i in neighbors:
            #     alpha.append(self.G.nodes[i]['alpha'])
            #     beta.append(self.G.nodes[i]['beta'])
        
            #single neighbor option
            neighbors = list(self.G.neighbors(node))
            alpha = []
            beta = []
            alpha.append(self.G.nodes[node]['alpha'])
            beta.append(self.G.nodes[node]['beta'])
            for i in neighbors:
                alpha.append(self.G.nodes[i]['alpha'])
                beta.append(self.G.nodes[i]['beta'])
                
            alpha = min(alpha) #sum(alpha)/len(alpha)
            beta = min(beta) #sum(beta)/len(beta)

            self.G.nodes[node]['alpha'] = alpha
            self.G.nodes[node]['beta'] = beta

            
        return
    
    def changeIDtoString(self):
        for node in list(self.G.nodes):
            mapping = {node: str(node)}
            self.G = nx.relabel_nodes(self.G, mapping)
        
        return
    
    def getGraph(self):
        return self.G #networkXgraph
    
    def getNodeInfo(self):
        nodes = list(self.G.nodes)
        #coords = np.empty(len(nodes)) #save in graph?
        for node in nodes:
            cloud = self.mapper.data_from_cluster_id(node, self.graph, self.data) #cloud associated to this node
            self.G.nodes[node]['coords'] = self.getCentroidCoords(cloud)
            (self.G.nodes[node]['x'], 
            self.G.nodes[node]['y'], 
            self.G.nodes[node]['z']) = self.getXYZfromCoords(self.G.nodes[node]['coords'])

            self.G.nodes[node]['alpha'] = self.getAlpha(cloud, self.G.nodes[node]['coords'])
            self.G.nodes[node]['beta'] = self.getBeta(cloud, self.G.nodes[node]['coords'])
            
        return
    
    def getCentroidCoords(self, cloud):
        cluster_coords = np.mean(cloud, axis = 0)
            
        return cluster_coords
    
    def getXYZfromCoords(self, coords):
        return coords[0], coords[1], coords[2]
        
    def getAlpha(self, cloud, node_coords):
        idx_z = 2
        center_z = node_coords[idx_z]
        cloud_z = cloud[:, idx_z]
        alpha = np.amax(np.absolute(center_z-cloud_z))
        
        return alpha
    
    def getBeta(self, cloud, node_coords):
        idx_y = 1
        center_y = node_coords[idx_y]
        cloud_y = cloud[:, idx_y]
        beta = np.amax(np.absolute(center_y-cloud_y))
        
        return beta

    def getLength(self):
        #loop over edges and take distance between nodes
        edges = list(self.G.edges)
        for edge in edges:
            node1_coords = self.G.nodes[edge[0]]['coords']
            node2_coords = self.G.nodes[edge[1]]['coords']
            #euclidiant distance
            dist = np.linalg.norm(node1_coords - node2_coords)
            self.G.edges[edge]['length'] = dist  
    
        return
    
    def removeIsolatedNodes(self):
        #remove outliers caused by voxel resolutions
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        
        return
    
    def getEdgeConnections(self): #dfn function

        pos = {}
        #get coords of all nodes an
        for node in list(self.G.nodes):  
            pos[node] = self.G.nodes[node]['coords']
        
        edge_connections = np.array([(pos[u], pos[v]) for u, v in list(self.G.edges)])

        return edge_connections
    
    def plotMapper(self):
        coords = self.getCoords()
        markersize = 3
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.data[:,0], self.data[:,1], self.data[:,2], s=markersize, alpha=0.1)
        ax.scatter3D(coords[:,0], coords[:,1], coords[:,2], c = 'r', s=markersize**4)
        
        edge_connections = self.getEdgeConnections()
        for vizedge in edge_connections:
            ax.plot(*vizedge.T, color="tab:gray", linewidth=4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

            
    def getCoords(self):
        nodes = list(self.G.nodes)
        coords = np.empty((len(nodes), 3))
        for i in range(len(nodes)):
            coords[i, :] = self.G.nodes[nodes[i]]['coords']
        
        return coords
    
    def decimatePoints(self, points, num_points):
        #should put this in utilities
        all_idx_random = np.random.permutation(len(points))
        idx_keep = all_idx_random[:num_points]
        points = points[idx_keep]
    
        return points
    
    def saveMapperView(self):
        # Visualize it
        self.mapper.visualize(self.graph, path_html="make_circles_keplermapper_output.html",
                  title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
        return
    
class graphMetrics:
    def __init__(self, nx_graph):
        edge_list, coords = self.getKarstNetInputsFromGraph(nx_graph)
        self.k = kn.KGraph(edge_list, coords)
        self.nx_graph = nx_graph

        #print('edge', edge_list)
        #print('dic', dic)
        #initiate karstnet object
            #Kgraph is class
            #kn or karstnet is file name
            #kgraph object constructor initializes reduced graph and graph with complete geometry and takes input:
                #list of edges
                #dictionary with keys as node names and value as coordinates
                #properties (optional dictionary with additional values associated with nodes)

    def getKarstNetInputsFromGraph(self, nx_graph):
        #input: networkx graph object
        #output: karstnet object (contains graph)
        edge_list = list(nx_graph.edges)
        node_list = list(nx_graph.nodes)
        coords = {}
        for node in node_list:
            #dic value has to be tuple
            coords[node] = tuple(nx_graph.nodes[node]['coords']) #just in case input is not tuple
        edge_list, coords = self.processKNinputs(edge_list, coords)
        return edge_list, coords 

    def processKNinputs(self, edge_list, coords):
        coords_proc = {int(key):value for key, value in coords.items()}
        edge_list_proc = [(int(edge[0]), int(edge[1])) for edge in edge_list]

        return edge_list_proc, coords_proc
                       
    def getMetrics(self):
        basic_metrics = self.k.basic_analysis()
        complex_metrics = self.k.characterize_graph(verbose=True)
        metric_dic = {**basic_metrics, **complex_metrics}

        return metric_dic
    
    def changeIDtoString(self, graph):
        #sloppy: copied from another written class
        for node in list(graph.nodes):
            mapping = {node: str(node)}
            graph = nx.relabel_nodes(graph, mapping)
        
        return graph
    
    def getLength(self, graph):
        #sloppy: copied from another written class
        #loop over edges and take distance between nodes
        edges = list(graph.edges)
        for edge in edges:
            node1_coords = graph.nodes[edge[0]]['coords']
            node2_coords = graph.nodes[edge[1]]['coords']
            #euclidiant distance
            dist = np.linalg.norm(node1_coords - node2_coords)
            graph.edges[edge]['length'] = dist  
    
        return graph
    
    def getGraphs(self):
        reduced_graph = self.changeIDtoString(self.k.graph_simpl)
        comp_graph = self.changeIDtoString(self.k.graph)
        
        return comp_graph, reduced_graph
    
    def getProcessedReducedGraph(self):
        reduced_graph = self.changeIDtoString(self.k.graph_simpl)
        reduced_graph = self.addGeometryToReducedGraph(reduced_graph)
        
        return reduced_graph
    
    def addGeometryToReducedGraph(self, reduced_graph):
        #copies geometry from mapper diss_graph to karstnet reduced graph so it can be used in diss code
        red_nodes = list(reduced_graph.nodes)
        for node in red_nodes:
            for key, value in self.nx_graph.nodes[node].items():
                reduced_graph.nodes[node][key] = value
        reduced_graph = self.getLength(reduced_graph)
        
        return reduced_graph
    
    #def get
        
    
    
class dataProcess:
    #def __init__(self):

    def pcUniformDensityXY(self, data, factor_x = 100, factor_y = 10, num_points = 100):
        #make pc density uniform in the x-axis
        #factor: 1) determines number of hypercubes to divide the x-space
                #2) determines +- range at which to look for points
        #num_points: number of maximum points to take from the hypercube 
        
        x = data[:, 0]
        y = data[:, 1]
    
        #x-domain boundary
        x_max = x.max()
        x_min = x.min()
        #y-domain boundary
        y_max = y.max()
        y_min = y.min()
        
        x_domain_size = abs(x_max) + abs(x_min)
        y_domain_size = abs(y_max) + abs(y_min)

        dx = x_domain_size/factor_x                 #determines +- range at which to look for points
        dy = y_domain_size/factor_y
        
        ref_x = np.arange(x_min, x_max + dx, dx)    #determines points in the middle of the range to look for points
        ref_y = np.arange(y_min, y_max + dy, dy)    #determines points in the middle of the range to look for points

        pc = np.zeros((1,3))
        for j in range(len(ref_y)):
            flag_y = np.logical_and(y > ref_y[j] - dy/2, y < ref_y[j] + dy/2)
            for i in range(len(ref_x)):
                flag_x = np.logical_and(x > ref_x[i] - dx/2, x < ref_x[i] + dx/2)
                flag = np.logical_and(flag_x, flag_y)
                points = data[flag,:]
                points = self.decimatePoints(points, num_points)
                pc = np.vstack((pc, points))
        pc = np.delete(pc, 0, 0)
    
        #potential errors
            #get domainsize from point cloud instead?
    
        return pc
    
    def pcUniformDensityX(self, data, factor = 100, num_points = 100):
        #make pc density uniform in the x-axis
        #factor: 1) determines number of hypercubes to divide the x-space
                #2) determines +- range at which to look for points
        #num_points: number of maximum points to take from the hypercube 
        
        #x-domain boundary
        x_max = data[:,0].max()
        x_min = data[:,0].min()
        
        domain_size = abs(x_max) + abs(x_min)
        dx = domain_size/factor                     #determines +- range at which to look for points
        
        ref_x = np.arange(x_min, x_max + dx, dx)    #determines points in the middle of the range to look for points
        x = data[:, 0]
        pc = np.zeros((1,3))
        for i in range(len(ref_x)):
            flag = np.logical_and(x > ref_x[i] - dx/2, x < ref_x[i] + dx/2)
            points = data[flag,:]
            points = self.decimatePoints(points, num_points)
            pc = np.vstack((pc, points))
        pc = np.delete(pc, 0, 0)
    
        #potential errors
            #get domainsize from point cloud instead?
    
        return pc
    
    def decimatePoints(self, points, num_points):
        #if no points then vstack after won't be affected!
        if len(points) < num_points:
            num_points = len(points)
        #should put this in utilities
        all_idx_random = np.random.permutation(len(points))
        idx_keep = all_idx_random[:num_points]
        points = points[idx_keep]

        return points
    
    def getDBSCANeps(self, pc):
        neigh = NearestNeighbors()
        nbrs = neigh.fit(pc)
        distances, indices = nbrs.kneighbors(pc)        
        d = np.sort(distances, axis=0)
        d = d[:,1]
        data = np.transpose(np.array([np.arange(1,len(d)+1), d]))
        rotor = Rotor()
        rotor.fit_rotate(data)
        elbow_index = rotor.get_elbow_index()
        eps = d[elbow_index]
        
        return eps
    
    def plotPC(self, data):
        markersize = 3
        ax = plt.axes(projection='3d')
        ax.scatter3D(data[:,0], data[:,1], data[:,2], s=markersize, alpha=0.1)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    
    def getData(self, num_files):
        #specific function to read data generated from graph diss models
        #computes connectivity 
        #returns dictionary with keys as metrics and values as all generated models
            #fracture dictionary
            #diss dictionary

        #num_files = 4
        
        base_path = '/Users/kanfar/Drive/Academic/PhD Research/Coding Library/Karst/crackDiss/graph metrics/too small/'
        
        frac_metrics = {}
        diss_metrics = {}
        
        #initialize
        add_path_diss = str(1) + '/' + 'diss_metrics.npy'
        path = base_path + add_path_diss
        
        data = np.load(path, allow_pickle=True)
        data = data[()]
        universal_keys = data.keys()
        for key in universal_keys:
            frac_metrics[key] = []
            diss_metrics[key] = []
            
        for i in range(1, num_files + 1):
            add_path_diss = str(i) + '/' + 'diss_metrics.npy'
            add_path_frac = str(i) + '/' + 'frac_metrics.npy'
        
            path_diss = base_path + add_path_diss 
            path_frac = base_path + add_path_frac 
        
            data_diss = np.load(path_diss, allow_pickle=True)
            data_frac = np.load(path_frac, allow_pickle=True)
        
            data_diss = data_diss[()]
            data_frac = data_frac[()]
            
            for key in data.keys():
                diss_metrics[key].append(data_diss[key])
                frac_metrics[key].append(data_frac[key])
            
            diss_metrics = self.calcConnectivity(diss_metrics)
            frac_metrics = self.calcConnectivity(frac_metrics)
    
        return diss_metrics, frac_metrics
    
        
    def calcConnectivity(self, dic):
        alpha = np.array(dic['alpha'])/0.25
        beta = (np.array(dic['beta'])-1)/0.5
        gamma = (np.array(dic['gamma'])-0.33)/0.17
        D = (alpha + beta + gamma)/3
        dic['connectivity'] = list(D)
    
        return dic
    
    def getArrayFromDic(self, dic, target_keys):
        #dic['correlation vertex degree'] = np.abs(dic['correlation vertex degree'])
        #assume dic values are all the same values
        arr = np.zeros((len(dic[target_keys[0]]), len(target_keys)))
        for i in range(len(target_keys)):
            arr[:, i] = dic[target_keys[i]]
            
        return arr


    def biKDE_2var(self, var1, var2, var3, d_obs):
        #number of plots is nC2 
        #compares the bivariate dist of two variables
        #plots bivariate of the features in var 1 as well as bivariate of features in var 2
        #var1 has num_var features = var 2
    
        #input must be array 
        #save correct headers now! headers
        #save points
        
        # Set up the figure
        num_var = var1.shape[1]    
        #num_var = 5
        fig = plt.figure()
        grid = plt.GridSpec(num_var, num_var, wspace=0.4, hspace=0.4) #0.75
        for i in range(1, num_var):
            for j in range(i):
                #print((num_var-1-i, num_var-1-j))
                fig.add_subplot(grid[i, j])
                
                shade_bool = True
                #sns.set_palette("pastel")
                sns.kdeplot(y = var3[:, num_var-1-i], x = var3[:, num_var-1-j], #label='literature')#,
                              fill=True, color = 'b')
                              #cmap="light:b", shade=shade_bool)
                              #cmap="Greens", shade=shade_bool)
                sns.kdeplot(y = var2[:, num_var-1-i], x = var2[:, num_var-1-j], #label='fracture')#,
                               fill=True, color = 'orange')
                #               #cmap="Blues", shade=shade_bool)
                sns.kdeplot(y = var1[:, num_var-1-i], x = var1[:, num_var-1-j], #label='simulation')#,
                              fill=True, color = 'g')
                              #cmap="Blues", shade=shade_bool)

                plt.scatter(d_obs[num_var-1-j], d_obs[num_var-1-i], marker = 'x', color ='k')
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                #plt.legend()
        return
    
    def uniKDE_2var(self, dic1, dic2, d_obs, header):
        width = 2
        sns.kdeplot(dic1[header], color ='g', linewidth=width)
        sns.kdeplot(dic2[header], color ='b', linewidth=width)
        #plt.axvline(d_obs[header], color = 'k' , linewidth=width)
        plt.xlabel(header)
        plt.legend(['Simulations', 'Literature'])#, 'Brejoes Cave'])
        
        return
    
    def uniKDE_all(self, dic1, dic2, dic3, d_obs, header):
        width = 2
        sns.kdeplot(dic1[header], color ='g', linewidth=width)
        sns.kdeplot(dic2[header], color ='b', linewidth=width)
        sns.kdeplot(dic3[header], color ='orange', linewidth=width)
        plt.axvline(d_obs[header], color = 'k' , linewidth=width)
        plt.xlabel(header)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.locator_params(axis='both', nbins=5)
        #plt.legend(['Simulations', 'Literature'])#, 'Brejoes Cave'])
        
        return
    
    def uniKDE_frac(self, dic1, dic2, header):
        width = 2
        sns.kdeplot(dic1[header], color ='g', linewidth=width)
        sns.kdeplot(dic2[header], color ='orange', linewidth=width)
        plt.xlabel(header)
        plt.legend(['Simulations', 'Initial Fracture'])
        
        return
    
    
    def uniKDE(self, dic1, header):
        width = 2
        fig, ax = plt.subplots()

        sns.kdeplot(dic1[header], color ='b', linewidth=width, ax = ax)
        ax2 = ax.twinx()
        sns.histplot(dic1[header], stat = 'count', bins = 10, ax = ax2)
        plt.xlabel(header)
        
        return
    
    
    
    # def biKDE_2var(self, var1, var2, var3, d_obs):
    #     #number of plots is nC2 
    #     #compares the bivariate dist of two variables
    #     #plots bivariate of the features in var 1 as well as bivariate of features in var 2
    #     #var1 has num_var features = var 2
    
    #     #input must be array 
    #     #save correct headers now! headers
    #     #save points
        
    #     # Set up the figure
    #     num_var = var1.shape[1]    
    #     #num_var = 5
    #     fig = plt.figure()
    #     grid = plt.GridSpec(num_var, num_var, wspace=0.25, hspace=0.25) #0.75
    #     for i in range(1, num_var):
    #         for j in range(i):
    #             #print((num_var-1-i, num_var-1-j))
    #             fig.add_subplot(grid[i, j])
                
    #             shade_bool = True
    #             #sns.set_palette("pastel")
    #             sns.kdeplot(x = var3[:, num_var-1-i], y = var3[:, num_var-1-j], #label='literature')#,
    #                           fill=True)
    #                           #cmap="light:b", shade=shade_bool)
    #                           #cmap="Reds", shade=shade_bool)
    #             sns.kdeplot(x = var2[:, num_var-1-i], y = var2[:, num_var-1-j], #label='fracture')#,
    #                           fill=True)
    #                           #cmap="Blues", shade=shade_bool)
    #             sns.kdeplot(x = var1[:, num_var-1-i], y = var1[:, num_var-1-j], #label='simulation')#,
    #                           fill=True)
    #                           #cmap="Greens", shade=shade_bool)

    #             plt.scatter(d_obs[num_var-1-i], d_obs[num_var-1-j],marker = 'x', color ='k')

    #     return
    
    def getBoersmaGraph(self):
        path = '/Users/kanfar/Drive/Academic/PhD Research/Coding Library/Karst/crackDiss/boersma Karst/'
        adj = scipy.io.loadmat(path + 'adj_TI_karst_20.mat')['adj_TI_karst_20']
        G = nx.from_numpy_matrix(adj)

        return G
    
    def getBoersmaGraphMetrics(self):
        G = self.getBoersmaGraph()
        node_lst = list(G.nodes)
        edge_lst = list(G.edges)

        path = '/Users/kanfar/Drive/Academic/PhD Research/Coding Library/Karst/crackDiss/boersma Karst/'
        coords = scipy.io.loadmat(path + 'coords_20.mat')['coords_20']
        coord_dic = {}
        for i in range(len(node_lst)):
            coord_dic[node_lst[i]] = tuple(coords[i,:])

        k = kn.KGraph(edge_lst, coord_dic)
        basic_metrics = k.basic_analysis()
        complex_metrics = k.characterize_graph(verbose=True)
        metric_dic = {**basic_metrics, **complex_metrics}
        
        alpha = metric_dic['alpha']/0.25
        beta =  (metric_dic['beta']-1)/0.5
        gamma = (metric_dic['gamma']-0.33)/0.17
        D = (alpha + beta + gamma)/3
        metric_dic['connectivity'] = D
                
        return metric_dic
    
    
    def getMetricsScalar(self, dic, target_keys):
        #dic['correlation vertex degree'] = np.abs(dic['correlation vertex degree'])
        #assume dic values are all the same values
        arr = np.zeros(len(target_keys))
        for i in range(len(target_keys)):
            arr[i] = dic[target_keys[i]]
            
        return arr
    
    def getPaulineMetrics(self):
        path = '/Users/kanfar/Drive/Academic/PhD Research/Coding Library/Karst/crackDiss/Literature Metrics/Pauline_metrics.npy'
        pauline_metrics = np.load(path)
        headers = ['nb_nodes', 'nb_edges', 'nb_connected_components',
                    'nb_cycles', 'alpha', 'beta', 'gamma', 'connectivity', 'global cyclic coef', 
                   'mean degree', 'cv degree', 'correlation vertex degree', 'aspl', 'cpd' ,
                   'total length', 'orientation entropy', 'nb_nodes_comp', 'nb_edges_comp', 
                   'mean length', 'length entropy', 'cv length', 'tortuosity']
        pauline_metric_dic = {}
        for j in range(pauline_metrics.shape[1]):
            pauline_metric_dic[headers[j]] = pauline_metrics[:, j]
        #process cv degree
        pauline_metric_dic['cv degree'] = pauline_metric_dic['cv degree']/100
        
        return pauline_metric_dic
    
    def getEdgeConnections_temp(self, G): #dfn function

        pos = {}
        #get coords of all nodes an
        for node in list(G.nodes):  
            pos[node] = G.nodes[node]['coords']
        
        edge_connections = np.array([(pos[u], pos[v]) for u, v in list(G.edges)])

        return edge_connections
    
    def plotMapper_temp(self, G, pc):
        pc = self.decimatePoints(pc, num_points = 5000)
        coords = self.getCoords_temp(G)
        markersize = 3
        ax = plt.axes(projection='3d')
        ax.scatter3D(pc[:,0], pc[:,1], pc[:,2], s=markersize, alpha=0.1)
        ax.scatter3D(coords[:,0], coords[:,1], coords[:,2], c = 'r', s=markersize**4)
        
        edge_connections = self.getEdgeConnections_temp(G)
        for vizedge in edge_connections:
            ax.plot(*vizedge.T, color="tab:gray", linewidth=4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xbound(lower = -3, upper = 3)
        ax.set_ybound(lower = -3, upper = 3)
        ax.set_zbound(lower = -0.7, upper = 0.7)
        
        
    def getCoords_temp(self, G):
        nodes = list(G.nodes)
        coords = np.empty((len(nodes), 3))
        for i in range(len(nodes)):
            coords[i, :] = G.nodes[nodes[i]]['coords']
        
        return coords
    ################################################

class vecTopoints:
    
    #code can be much more elegant

    def __init__(self):
        return

    def irregular(self, startPoint, endPoint, space):
        seg = np.diff(space)
        #num_points = len(seg) - 1
        num_points = len(seg) + 1
        
        #points = np.empty((num_points +2 , 3))
        points = np.empty((num_points , 3))
        
        points[0,:] = startPoint
        points[-1,:] = endPoint #redundanet. next point already reaches the last point
    
        azimuth, inclination = self.getDirection(startPoint, endPoint)
        temp = np.copy(startPoint)
        for i in range(num_points - 2): #previously just num_points (no need to compute last point because given by end point above)
            #i = i + 1
            temp = self.nextPoint(seg[i], azimuth, inclination, temp, startPoint, endPoint)
            points[i + 1, :] = temp
        
        return points, azimuth, inclination
    
    # def regular(self, startPoint, endPoint, num_points):
    #     #fix direction: endPoint < startPoint doesn't work
        
    #     dist_x = abs(startPoint[0]-endPoint[0])
    #     dist_y = abs(startPoint[1]-endPoint[1])
    #     dist_z = abs(startPoint[2]-endPoint[2])
        
    #     interval_x = dist_x/(num_points + 1)
    #     interval_y = dist_y/(num_points + 1)
    #     interval_z = dist_z/(num_points + 1)
        
    #     points = np.empty((num_points +2 , 3))
    #     points[0,:] = startPoint
    #     points[-1,:] = endPoint
    
    #     for i in range(num_points):
    #         i = i + 1
    #         points[i, 0] = startPoint[0] + interval_x*i
    #         points[i, 1] = startPoint[1] + interval_y*i
    #         points[i, 2] = startPoint[2] + interval_z*i
            
    #     return points
    
    def regular(self, startPoint, endPoint, num_points):
        #num_points is in between points
    #fix direction: endPoint < startPoint doesn't work
    
        dist_x = abs(startPoint[0]-endPoint[0])
        dist_y = abs(startPoint[1]-endPoint[1])
        dist_z = abs(startPoint[2]-endPoint[2])
        
        interval_x = dist_x/(num_points + 1)
        interval_y = dist_y/(num_points + 1)
        interval_z = dist_z/(num_points + 1)
        
        points = np.empty((num_points + 2 , 3))
        points[0,:] = startPoint
        points[-1,:] = endPoint
    
        for i in range(num_points):
            i = i + 1
            if endPoint[0] > startPoint[0]:
                points[i, 0] = startPoint[0] + interval_x*i
            elif endPoint[0] == startPoint[0]:
                points[i, 0] = startPoint[0] 
            else:
                points[i, 0] = startPoint[0] - interval_x*i
                
            if endPoint[1] > startPoint[1]:
                points[i, 1] = startPoint[1] + interval_y*i
            elif endPoint[1] == startPoint[1]:
                points[i, 1] = startPoint[1] 
            else:
                points[i, 1] = startPoint[1] - interval_y*i
                
            if endPoint[2] > startPoint[2]:
                points[i, 2] = startPoint[2] + interval_z*i
            elif endPoint[2] == startPoint[2]:
                points[i, 2] = startPoint[2] 
            else:
                points[i, 2] = startPoint[2] - interval_z*i
            
        return points
            
    
    def getDirection(self, startPoint, endPoint):
        vec = abs(endPoint - startPoint)
        #vec = abs(endPoint - startPoint)

        #using spherical coordinates
        azi = self.getAzimuth(vec)
        inc = self.getInclination(vec)
        
        return azi, inc
    
    def getInclination(self, vec):
        #calc vector displacement/magnitude
        a = np.copy(vec)
        #calc vector projection to xy plane magnitude
        b = np.copy(vec)
        b[2] = 0
        if np.linalg.norm(b) == 0:
            rad = 0
        else:
        #calc angle between vector and projection
            rad = self.getAngle(a,b)
            rad = np.pi/2 - rad #angle from z
        
        return rad
    
    # def getInclination(self, vec):
    #     #old correct
    #     #calc vector displacement/magnitude
    #     r = np.linalg.norm(vec)
    #     #calc vector projection to xy plane magnitude
    #     temp = np.copy(vec)
    #     temp[2] = 0
    #     side = np.linalg.norm(temp)
    #     #calc angle between vector and projection
    #     cos = side/r
    #     rad = np.arccos(cos)
    #     rad = np.pi/2 - rad
        
    #     return rad
    
    def getAzimuth(self, vec):
        #calculates angle from the unit vector x (1,0,0), clockwise or counter doesn't matter
        v = np.copy(vec)
        #project z to xy plane
        v[2] = 0
        #project vector to x
        x_unit = np.array([1, 0, 0])
        #special case
        if np.linalg.norm(v) == 0:
            rad = 0
        #find angle between x and vector in xy plane (angle between x and y)
        else: 
            rad = self.getAngle(x_unit, v)
        
        return rad
    
    def getAngle(self, vec1, vec2):
        
        # inner = np.inner(vec1, vec2) #just take the x value
        # norms = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        # cos = inner/norms
        
        #if only z component or other spectial cases
        #if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        #    inner = 0
        #else: 
        norm_vec1 = vec1/np.linalg.norm(vec1)
        norm_vec2 = vec2/np.linalg.norm(vec2)
        inner = np.inner(norm_vec1, norm_vec2)
        rad = np.arccos(inner)
        
        return rad
    
    def convertAngles(self, azi, inc, startPoint, endPoint):
        #azimuth
        if startPoint[1] > endPoint[1]:
            #check direction is towards +x or -x (given points are based on ordered edges)
            if startPoint[0] < endPoint[0]: #->#
                azi = -azi
            else:
                azi = -azi
                
        elif startPoint[1] < endPoint[1]:
            if startPoint[0] < endPoint[0]: #->#
                azi = azi
            else:
                azi = -azi
            
                
            
                
            
        return
            
        
    
    def nextPoint(self, r, azi, inc, prevPoint, startPoint, endPoint):
        
        x = r*np.cos(azi)*np.sin(inc)
        y = r*np.sin(azi)*np.sin(inc)
        z = r*np.cos(inc)
        
        #if endpoint == start point no need to keep doing the same update in that direction every new point
        
        #add = np.array([x, y, z])
        
        if endPoint[0] > startPoint[0]:
            x_next = prevPoint[0] + x
        elif endPoint[0] == startPoint[0]:
            x_next = startPoint[0]
        else:
            x_next = prevPoint[0] - x
            
        if endPoint[1] > startPoint[1]:
            y_next = prevPoint[1] + y
        elif endPoint[1] == startPoint[1]:
            y_next = startPoint[1]
        else:
            y_next = prevPoint[1] - y
            
        if endPoint[2] > startPoint[2]:
            z_next = prevPoint[2] + z
        elif endPoint[2] == startPoint[2]:
            z_next = startPoint[2]
        else:
            z_next = prevPoint[2] - z
            
            
        nextPoint = np.array([x_next, y_next, z_next])
        
        return nextPoint
    

    def plot(self, points):
        plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        ax.scatter3D(points[:,0], points[:,1], points[:,2])
    