# Dynamic Graph Dissolution

Dynamic Graph Dissolution is an algorithm for telogenetic karst evolution. The algorithm models conduit evolution using graph representations of discrete fracture networks that dynamically update during dissolution. Here is a Python implementation of DGD. 

![workflow_paper](https://user-images.githubusercontent.com/47835011/206561361-e2ae75a2-9d2a-4214-b822-73ccaa175767.jpeg)

> Title: Stochastic Geomodelling of Karst Morphology by Dynamic Graph Dissolution

> Authors: Rayan Kanfar · Tapan Mukerji

> Abstract: Cave networks are excellent groundwater and hydrocarbon reservoirs. Cave geometry, spatial distribution, and interconnectivity is critical for developing production and contaminant remediation strategies. Geologically realistic stochastic models for simulating karst are essential for quantifying the spatial uncertainty of karst networks given geophysical observations. Dynamic Graph Dissolution, a novel physics-based approach for three-dimensional stochastic geomodelling of telogenetic karst morphology is introduced. The cave evolution is modelled through dissolution of fractures over geologic time based on a graph representation of discrete fracture networks, which can be informed by field observations. The graph is initially modelled based on fracture intersections. In order to account for overlapping enlargements, the graph representation is updated over dissolution using the Mapper algorithm with density-based spatial clustering. This modelling approach enables generation of multiple realizations of different geologic scenarios of karst formation at a tractable computational cost. Realizations generated using the proposed algorithm are compared with real caves using graph topological metrics such as central point dominance, connectivity, average degree, degree dispersion, and assortativity. The distributions of graph topological metrics of generated realizations overlap with the metrics of known caves suggesting that the graph structure of observed and simulated caves are at least globally similar.

For any question, please contact [kanfar@stanford.com]

# Licenses 
All material is made available under MIT license.

# Code
Please follow the crack dissolution and dynamic graph dissolution notebooks. 
