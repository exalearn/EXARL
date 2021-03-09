import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from collections import Counter
import logging

def parse_structure(file_name):
    '''parse the database xyz files
       1) a list of the coordinates of each cluster (returns 1st)
          ['O   1.922131  -1.3179458  -2.891314\n', 'H   1.9396669  -1.812519  -2.0610204\n', ...]
       2) the energy (returns 2nd)'''
    with open(file_name) as f:
        n_atoms = f.readline()       #atoms per cluster
        n_lines = 2 + int(n_atoms)   #lines per cluster (#atoms + energy + coords)
    with open(file_name) as f:
        lines = f.readlines()
    energies = np.array(lines[1::n_lines],dtype='float32') 
    structure_list, energy_list = [], []
    for n in range(int(energies.shape[0])):
        structure_list.append(lines[n_lines*(n+1)-n_lines:n_lines*(n+1)])
        structure_list[n][1]=float(structure_list[n][1])  #energy in float
        energy_list.append(float(structure_list[n][1]))
    return structure_list, energy_list


def adjacency_list(coord_list):
    '''Creates adjacency list to form graph in graph_analytics.
       Structure list must be ordered such that the two covalently
       bound hydrogens directly follow their oxygen.
       Definition of a hydrogen bond obtained from
       https://aip.scitation.org/doi/10.1063/1.2742385'''
    s = pd.Series(coord_list)
    dfx=s.str.split(expand=True)
    coords=[str.split(x) for x in coord_list]
    
    #delete atom label
    for j in coords:
        del j[0]

    cov_bonds, h_bonds, labels = [],[],[]
    for i, row in dfx.iterrows():
        labels.append([str(i+1),row[0]])
    q_1_2=[]
    for i in range(len(dfx)):
        if s[i].split()[0]=='O':
            cov_bonds.append([str(i+1),str(i+2),'covalent'])
            cov_bonds.append([str(i+1),str(i+3),'covalent'])
            h1=np.array(s[i+1].split()[1:],dtype='float64')
            h2=np.array(s[i+2].split()[1:],dtype='float64')
            o=np.array(s[i].split()[1:],dtype='float64')
            q_1_2.append([h1-o, h2-o])
    v_list=[np.cross(q1,q2) for (q1,q2) in q_1_2]
    for idx, v in enumerate(v_list):
        for index, both_roh in enumerate(q_1_2):
            for h_index, roh in enumerate(both_roh):
                indexO=((idx+1)*3)-2
                indexH=((index+1)*3)-2+(h_index+1)
                o_hbond = s[indexO-1].split()
                try:
                    h_hbond= s[indexH-1].split()  #will enumerate past the list if you let it
                except KeyError:
                    continue
                dist = np.linalg.norm(np.array(o_hbond[1:],dtype='float64')-np.array(h_hbond[1:],dtype='float64'))
                if (dist>1) & (dist<2.8):
                    angle = np.arccos(np.dot(roh, v)/(np.linalg.norm(roh)*np.linalg.norm(v)))*(180.0/np.pi)
                    if angle > 90.0:
                        angle=180.0-angle
                    N = np.exp(-np.linalg.norm(dist)/0.343)*(7.1-(0.05*angle)+(0.00021*(angle**2)))
                    if N >=0.0085:
                        h_bonds.append([str(indexO),str(indexH),'hydrogen'])

    return labels, cov_bonds, h_bonds, coords


def load_graph(struct):
    '''loads the graph for a single structure and returns the graph'''
    l,c,h,coords=adjacency_list(np.array(struct))
    node_labels = dict()
    for i in range(len(l)):
        node_labels[l[i][0]] = l[i][1]
    edges=c+h
    graph = nx.Graph()
    for k,v in node_labels.items():
        graph.add_node(k, label=v, coords=coords[int(k)-1])
    for triple in edges:
        atom1 = [float(x) for x in coords[int(triple[0])-1]]
        atom2 = [float(x) for x in coords[int(triple[1])-1]]        
        distance = np.linalg.norm(np.array(atom2)-np.array(atom1))
        graph.add_edge(triple[0], triple[1], label=triple[2], weight=distance)
    return graph, node_labels, edges


def graph_projection(graph):
    """
    Project a new graph based on connectivity between a subset of nodes in the
    input graph (subsequently called "targets") via another subset (non-overlapping
    with the targets) of nodes referred to as "connectors". Connectors can be
    the complementary set of targets, aka set(V(graph) - targets) or a subset of that
    complementary set.
    """
    targets = [node for node,attr in graph.nodes(data=True) if attr['label']=='O']
    connectors = set([node for node,attr in graph.nodes(data=True) if attr['label']=='H'])
    weighted_edges = Counter()

    for node1 in connectors:
        neighbors = graph[node1]
        neighboring_targets = []
        for node2 in neighbors:
            if node2 in targets:
                neighboring_targets.append(node2)
        # iterate over the list of all candidate nodes that are connected to
        node_count = len(neighboring_targets)
        for i in range(node_count-1):
            for j in range(i+1, node_count):
                weighted_edges[(neighboring_targets[i], neighboring_targets[j])] += 1
    return weighted_edges


def molecular_role_distribution(graph):
    """
    Function that computes a donor-acceptor profile for each water molecule.
    It accepts a graph with hydrogen and oxygen atoms as nodes, and returns
    two dictionaries.
    The first dictionary provides molecule-level role description. Keys in
    the dictionary correspond to the oxygen atom ids in each water molecule.
    The values are the corresponding donor-acceptor roles aggregated from the
    hydrogen and oxygen atoms in that molecule.  Profiles such as
    donor-acceptor-donor or donor-donor-acceptor are both represented as a1d2.
    The second dictionary is a Counter that aggregates the counts from
    the distribution stored in the first dictionary.
    Example: Counter({'a2d2': 12, 'a1d2': 9, 'a2d1': 8, 'a3d2': 1})
    """
    atomic_roles = defaultdict(Counter)
    node_labels = nx.get_node_attributes(graph, 'label')
    # Iterate over all hydrogen bonds.  Assign the hydrogen atom a "donor"
    # role and the oxygen atom an "acceptor" role.
    for edge in list(graph.edges(data=True)):
        if edge[2]['label'] == 'hydrogen':
            role = 'd' if node_labels[edge[0]] == 'H' else 'a'
            atomic_roles[edge[0]][role] += 1
            role = 'd' if node_labels[edge[1]] == 'H' else 'a'
            atomic_roles[edge[1]][role] += 1

    # Iterate over all oxygen atoms. For each oxygen atom, aggregate the
    # role counts from each of the hydrogen atoms connected to it via a
    # covalent bond. Thus each oxygen atom aggregates the roles for the
    # water molecule it is part of.
    molecular_roles = dict()
    molecular_role_distribution = Counter()
    for node in list(graph.nodes(data=True)):
        # Iterate over a list such as
        # [('24', {'label': 'H'}), ('25', {'label': 'O'})
        # Only do the computation of nodes representing oxygen atoms
        node_id = node[0]
        node_label = node[1]['label']
        if node_label != 'O':
            continue
        neighbors = graph[node_id]
        for n in neighbors:
            if neighbors[n]['label'] == 'covalent':
                roles = atomic_roles[n]
                atomic_roles[node_id] += roles
        tmp_list = []
        for k in sorted(atomic_roles[node_id].keys()):
            tmp_list.append('%s%d' % (k, atomic_roles[node_id][k]))
        role_key = ''.join(tmp_list)
        molecular_roles[node_id] = role_key
        molecular_role_distribution[role_key] += 1
    return molecular_roles,molecular_role_distribution


def get_directed_coarse_graph(graph, edge_weight=2.90):
    node_roles, role_distribution = molecular_role_distribution(graph)
    weighted_edges = graph_projection(graph)

    dc_graph = nx.DiGraph()
    for k,v in node_roles.items():
        dc_graph.add_node(k, label=v)
    for edge, count in weighted_edges.items():
        dc_graph.add_weighted_edges_from([(edge[0], edge[1], edge_weight)])

    attrs=nx.get_edge_attributes(dc_graph,'weight')

    for key in attrs:
        attrs[key]={"length": attrs[key]}

    nx.set_edge_attributes(dc_graph, attrs)

    return dc_graph


def get_ring_key(list_of_nodes):
    """
    Learn unique representations of a ring chain.
    :param list_of_nodes: a sequence of nodes that captures a ring
                          Example: [4, 1, 5, 4]
    :return a string representation of the path -> "1,4:1,5:4,5"
    """

    # Given a list of nodes V = [list_of_nodes[i-1], list_of_nodes[i]]
    # ','.join(sorted(list_of_nodes[i-1], list_of_nodes[i])) returns a
    # string of form "v1, v2" such that v1 = V_sorted[0] and v2  = V_sorted[1]

    list_of_ordered_edges = [','.join(sorted([str(list_of_nodes[i-1]), \
            str(list_of_nodes[i])])) \
            for i in range(1, len(list_of_nodes))]

    # For above input list list_of_ordered_edges = [(1,4), (1,5), (4,5)]
    # Next, sort the ordered edges and build a string of form "e1:e2:..:e_n" where
    # the e_i is encoded by the above statement.
    # ring_key for above example: "1,4:1,5:4,5"

    ring_key = ':'.join(sorted(list_of_ordered_edges))
    return ring_key


def find_rings(graph, ring_size, node=-1):
    """
    Find the number of rings associated with a node via depth-first-search.
    :param graph: Input graph to count rings in.
    :param node: Count number of rings attached to this node. Count rings in 
                 whole graph is node is set to -1.
    :param ring_size: Size of the ring, would be 3 for a trimer.
    :return Number of rings found
    """

    # Initialize visit stack.  We need a first-in-last-out data structure 
    # for depth-first-search (DFS). The stack will contain a list of paths that
    # will be expanded in a DFS fashion.
    visit_stack = [[n] for n in graph.nodes()] if node == -1 else [[node]]
    # Track visited nodes so we don't fall in a loop
    visited = set()
    rings = [] 
    ring_key_set = set()
   
    while len(visit_stack) > 0:
        path_to_expand = visit_stack.pop()
        next_node = path_to_expand[-1]
        curr_path_len = len(path_to_expand)

        for nbr in graph.neighbors(next_node):
            if curr_path_len == ring_size:
                if nbr == path_to_expand[0]:
                    ring_path = path_to_expand + [nbr]
                    ring_key = get_ring_key(ring_path)
                    G_sub=graph.subgraph(ring_path)
                    degrees=[row[1] for row in G_sub.degree()]
                    if 2 == np.mean(degrees):
                    # Check if this ring is already found.  
                    # For example, assumg we start the search from 
                    # node 0 in the test graph.  We should not double
                    # count the rings (0->1->3->2) and (0->2->3->1)
                        if ring_key not in ring_key_set:
                            rings.append(path_to_expand)
                            ring_key_set.add(ring_key)

            elif curr_path_len < ring_size:
                # Expand this path further provided there is no loop
                if nbr not in path_to_expand:
                    new_path_to_expand = path_to_expand + [nbr]
                    visit_stack.append(new_path_to_expand)        

    return rings


def disconnection_check(coarse_graph):
    '''
    Check for disconnected water molecules by examining
    the node degrees in the coarse graph 
    '''
    disconnected_nodes = [x for x in coarse_graph.degree() if x[1]==0]
    if disconnected_nodes == []:
        return 0
    else:
        logging.info(f"The following nodes are disconnected: {[x[0] for x in disconnected_nodes]}")
        return len(disconnected_nodes)

def score_connectivity(coarse_graph):
    '''
    Computes a score based on the degree and variance in degree
    Optimizes toward increased connectivity + symmetry
    '''
    node_degrees_list = [x[1] for x in coarse_graph.degree()]
    return np.mean(node_degrees_list) - np.std(node_degrees_list)


def score_cycles(coarse_graph, cycle_size=6):
    return len(find_rings(coarse_graph, cycle_size))
