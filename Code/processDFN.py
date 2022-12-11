"""
Created on Thu Aug 26 14:19:22 2021

@author: kanfar
"""
import os 
import re
import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

class dfn:

    round_dec = 2

    def __init__(self, path):
        
        self.poly = path + 'polygons.dat'
        self.alpha = path + 'aperture.dat'
        self.nodeCoords = path + 'intersection_list.dat'
        self.graph = path + 'graph.gml'
        self.domain = path + 'params.txt'
        self.getPolygonCoords() 
        self.processGraph()

        return
        
    def getGraph(self):
        return self.G

    def processGraph(self):
        #load graph
        self.G = nx.read_gml(self.graph)
        self.getAperture()
        self.setAlphaToGraph()
        self.setBetaToGraph()
        self.setCoordstoNodes()
        #self.removeOverlapEdges()
        if nx.is_connected(self.G) == False:
            print('processed graph is not connected')

        return
    
    def setCoordstoNodes(self):
        temp = self.G.copy()

        temp.remove_node('s')
        temp.remove_node('t')

        for node in list(temp.nodes):
            x, y, z = self.getCoordsfromDic(node)
            self.G.nodes[node]['coords'] = np.stack((x,y,z))#(x,y,z)
        
        return

    def getAperture(self):
        file = open(self.alpha)         
        next(file)  
        lines = file.read()
        file.close()
        doc = lines.split()      
        file_array = np.array(doc)
                 
        idx_spacing = 4     #from DFN data file
        idx_init = 3        #from DFN data file
        idx = np.arange(idx_init, len(doc), idx_spacing)
        self.aperture = np.array(file_array[idx])
        
        return 
    
    def setAlphaToGraph(self):
        #alpha is aperture of fracture
        #double check if implementation is correct
        #since aperture is stochastic or constant, order doesn't matter but better be consistent
        #this is not necessary for fracture graphs since the sequence will be linear but will add the aperture correctly nonetheless
        edge_key = list(self.G.edges)
        for key in edge_key:
            frac_num = self.G.edges[key]['frac']
            if type(frac_num) == str:               #this is when it's the edges are connected from wall to source or target
                self.G.edges[key]['alpha'] = 0
            else:
                frac_num = frac_num - 1             #-1 because fracture labels start from 1 while aperture list starts from 0
                self.G.edges[key]['alpha'] = float(self.aperture[frac_num])
    
        return

    def getWidthIntersect(self):
        #assumption starting edge node is never 't' or 's'. if not case, fix.
        edge_key = list(self.G.edges)
        for key in edge_key:
            if key[1] == 's' or key[1] == 't':
                length = self.G.nodes[key[0]]['length']
                self.G.edges[key]['beta'] = length
            else:
                intersect_length1 = self.G.nodes[key[0]]['length']
                intersect_length2 = self.G.nodes[key[1]]['length']
                mean_length = (intersect_length1 + intersect_length2)/2
                self.G.edges[key]['beta'] = mean_length
            
        return 
    
    def setBetaToGraph(self):
        #only for intersection graph
        if self.G.graph['representation'] == 'intersection':
            self.getWidthIntersect()
        else:
            self.getWidthIntersect()
        
        return 
    
    def getPolygonCoords(self):

        #Can be improved. Problem: too customized for DFN file. 
        #function: read polygon file and change it into a list of numpy arrays
        #input: polygon.dat
        #output: list of numpy arrays. numpy array is a fracture containing numpy arrays of coordinates for the polgyon. 

        non_poly = 5                                #max number of characters in the delimiter 
        file = open(self.poly)         
        next(file)                                  #skip first line
        lines = file.read()
        file.close()
        doc = re.split(' {|}',lines)                #remove some delimeters 
        str_list = list(filter(None, doc))          #filter empty elements
        str_list.pop()                              #remove last element in list which is double space in polygons.dat file

        #change string list of coordinates for each fracture into elements in numpy array
        lst = []
        counter = -1 
        i = 0
            
        while i < len(str_list):
            if len(str_list[i]) < non_poly:
                counter = counter + 1
                i = i + 1
                lst.append([])
            lst[counter].append(np.asarray(re.split(',',str_list[i]), 'float'))
            i = i + 1
    
        self.polyCoords = lst
    
        return 
        
    def getCoordsfromDic(self, keys):
        x = self.G.nodes[keys]['x']
        y = self.G.nodes[keys]['y']
        z = self.G.nodes[keys]['z']
        return x, y, z
    
    def getGraphCoords(self):
        #get all graph coords except s and t (which don't have coords)
        temp = self.G.copy()

        temp.remove_node('s')
        temp.remove_node('t')

        pos = {}
        #get coords of all nodes an
        for keys in list(temp):  
            x, y, z = self.getCoordsfromDic(keys)
            #z = 0 ###
            nodeCoords = np.stack((x,y,z))
            pos[keys] = nodeCoords
        
        node_coords = np.array([pos[v] for v in list(temp.nodes)])
        edge_connections = np.array([(pos[u], pos[v]) for u, v in list(temp.edges)])

        return node_coords, edge_connections
    
    def getBoundaryNodes(self):
        #get boundary node coords 
        source_pos = {}
        source_neighbors = [neighbor for neighbor in nx.neighbors(self.G, 's')]
        for keys in source_neighbors:  
            x, y, z = self.getCoordsfromDic(keys)
            nodeCoords = np.stack((x,y,z))
            source_pos[keys] = nodeCoords
        source_coords = np.array([source_pos[v] for v in source_neighbors])

        
        target_pos = {}
        target_neighbors = [neighbor for neighbor in nx.neighbors(self.G, 't')]
        for keys in target_neighbors:  
            x, y, z = self.getCoordsfromDic(keys)
            nodeCoords = np.stack((x,y,z))
            target_pos[keys] = nodeCoords
        target_coords = np.array([target_pos[v] for v in target_neighbors])
        
        return source_coords, target_coords
    
    def getDomainSize(self):
        file = np.genfromtxt(self.domain)
        domain_size = file[-3:]
        
        return domain_size 
    
    def removeOverlapEdges(self):
        #get a list of multiple edges on the same fractures
        #get all edges on every fracture (fracture is each polygon)
            #if there's more than one edge on one fracture then one of them must be engulfing the others
            #remove engulfing edge
        
        #returns list of lists: each list is a fracture polygon and it contains a list of all edges in that fracture
        frac_list = self.getEdgesOnSameFrac()
        #returns a list of engulfing edges for each fracture
        redundant_edges = self.getRedundantEdges(frac_list)
        self.removeEdges(redundant_edges)
        return frac_list
    
    def getNumPolygons(self):
        num_frac = len(self.polyCoords)
        return num_frac
    
    def getEdgesOnSameFrac(self):
        #returns list of lists: each list is a fracture polygon and it contains a list of all edges in that fracture
        frac_list = []
        num_frac = len(self.polyCoords)
        
        frac = 1
        while frac <= num_frac:
            edge_list = []
            for edge in list(self.G.edges):
                if self.G.edges[edge]['frac'] == frac:
                    edge_list.append(edge)
            if len(edge_list) > 1: #why
                frac_list.append(edge_list)
            frac = frac + 1
    
        return frac_list
    
    
    def getRedundantEdges(self, frac_list):
        #rounding disclaimer
        redundant_list = []
        for edge_list in frac_list:
            #edge_list are all edges in a single fracture
            all_edge_comb_list = self.getAllEdgeCombinations(edge_list)
            all_length_comb_list = self.getAllEdgeCombinationLengths(all_edge_comb_list)
            redundant_list.extend(self.getEdgesToRemove(all_length_comb_list, edge_list))

        return redundant_list
    

    def getAllEdgeCombinations(self, edge_list):
        #all combination of edges of 2 edges and above
        edge_combinations = sum([list(map(list, combinations(edge_list, i))) for i in range(2, len(edge_list) + 1)], [])
        
        return edge_combinations
    
    def getAllEdgeCombinationLengths(self, edge_list):
        length_lst = []
        for lst in edge_list:
            total_sum = 0
            for edge in lst:
                total_sum = total_sum + self.G.edges[edge]['length']
            length_lst.append(total_sum)
        length_lst = set(length_lst)
        length_lst = list(length_lst)
        
        return length_lst
    
    
    def getEdgesToRemove(self, length_lst, edge_list):
        redundant_lst = []
        all_edge_lengths = []
        for edge in edge_list:
            for length in length_lst:
                if np.round(self.G.edges[edge]['length'], self.round_dec) == np.round(length, self.round_dec):
                    redundant_lst.append(edge)

            all_edge_lengths.append(self.G.edges[edge]['length'])
        redundant_lst = set(redundant_lst) #because of rounding, one edge can be same length as many lengths and is readded to the list
        redundant_lst = list(redundant_lst)
        self.checkRedundantEdges(edge_list, redundant_lst, all_edge_lengths, length_lst)
        
        return redundant_lst
    
    def checkRedundantEdges(self, edge_list, redundant_lst, all_edge_lengths, length_lst):
        #remove redundant edges from the fracture
        for edge in redundant_lst:
            edge_list.remove(edge) 
        
        #add all lengths of kept edges
        sum_keep_list = 0 
        for edge in edge_list:
            sum_keep_list = sum_keep_list + self.G.edges[edge]['length']
           
        #kept edges must add to total fracture length (max)
        max_length = max(all_edge_lengths)
        if np.round(max_length, self.round_dec) != np.round(sum_keep_list, self.round_dec):
            #print('comb lengths', length_lst)
            print('redund. edges', redundant_lst)
            print('all lengths', all_edge_lengths)
            print('kept edges', edge_list)
            print('error processing graph: non-overlapping edge removed')
        
        return
    
    def removeEdges(self, edges_list):
        for edge in edges_list:
            self.G.remove_edge(edge[0], edge[1])
        
        return
    
    def plotDFNbed(self, save):
        #if bedding plane is fracture

        fig = plt.figure()
        ax = a3.Axes3D(fig)
        ax = self.format_axes(ax)
        
        tri = a3.art3d.Poly3DCollection([self.polyCoords[0]])
        #tri.set_color(colors.rgb2hex(np.random.rand(3)))
        tri.set_facecolor('k')
        tri.set_edgecolor('k')
        tri.set_alpha(0.1)
        ax.add_collection3d(tri)
        for i in range(1, len(self.polyCoords)):
            tri = a3.art3d.Poly3DCollection([self.polyCoords[i]])
            tri.set_color(colors.rgb2hex(np.random.rand(3)))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
        plt.show()
        
        if save == 1:
            image_name = 'dfn.eps'

            image_format = 'eps'  # e.g .png, .svg, etc.
            fig.savefig(image_name, format=image_format, dpi=1200)
            
            
    def viewDFN(self, save = 0):
        
        fig = plt.figure()
        ax = a3.Axes3D(fig)
        ax = self.format_axes(ax)
        
        for i in range(len(self.polyCoords)):
            tri = a3.art3d.Poly3DCollection([self.polyCoords[i]])
            tri.set_color(colors.rgb2hex(np.random.rand(3)))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
        plt.show()
        
        if save == 1:
            image_name = 'dfn.eps'

            image_format = 'eps'  # e.g .png, .svg, etc.
            fig.savefig(image_name, format=image_format, dpi=1200)
    
    def viewGraph(self, save = 0):

        fig = plt.figure()
        ax = a3.Axes3D(fig)
        
        #this is because target and sources are added at the end of the node list
        node_coords, edge_connections = self.getGraphCoords()
        source_node, target_node = self.getBoundaryNodes()
        #node_coords[:,2] = 0
        #source_node[:,2] = 0 
        #target_node[:,2] = 0
        ax.scatter(*node_coords.T, s=100, ec="w")
        ax.scatter(*source_node.T, s=150, color="red")
        ax.scatter(*target_node.T, s=150, color="blue")
        
        # Plot the edges
        for vizedge in edge_connections:
            ax.plot(*vizedge.T, color="tab:gray")
        
        ax = self.format_axes(ax)     
        plt.show()
        
        if save == 1:
            image_name = 'graph.eps'

            image_format = 'eps'  # e.g .png, .svg, etc.
            fig.savefig(image_name, format=image_format, dpi=1200)
        return
    
    def format_axes(self, ax):
        """Visualization options for the 3D axes."""
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        domain_size = self.getDomainSize()
        
        xmin = -0.5*domain_size[0]
        xmax = 0.5*domain_size[0]
        ymin = -0.5*domain_size[1]
        ymax = 0.5*domain_size[1]
        zmin = -0.5*domain_size[2]
        zmax = 0.5*domain_size[2]
        
        xticks = np.linspace(xmin, xmax, 5)
        yticks = np.linspace(ymin, ymax, 5)
        zticks = np.linspace(zmin, zmax, 5)
        
        ax.set_xbound(lower = xmin, upper = xmax)
        ax.set_ybound(lower = ymin, upper = ymax)
        ax.set_zbound(lower = zmin, upper = zmax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks)
    
        return ax




import kmapper as km

class processMapper:
    
    #processMapper
    def __init__(self):
        self.mapper = km.KeplerMapper(verbose=2)
        return
    
    def getMapperCentroid(self, graph, data):
        cluster_ids = self.getClusterIDs(graph)
        cluster_coords = np.empty((len(cluster_ids), 3))
        for i in range(len(cluster_coords)):
            cloud = self.mapper.data_from_cluster_id(cluster_ids[i], graph, data)
            cluster_coords[i, :] = np.mean(cloud, axis = 0)
            
        return
    
    def getClusterIDs(self, graph):
        cluster_ids = list(graph['nodes'].keys())
        
        return cluster_ids
