import os
import gym
import numpy as np
import igraph
import random
import json
from collections import OrderedDict
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm

class ScriptEnv(gym.Env):

    def __init__(self):
        super().__init__()
        
        self.dir = ExaGlobals.lookup_params('dir')
        nodes = ExaGlobals.lookup_params('nodes')
        regularity = ExaGlobals.lookup_params('regularity')
        population = ExaGlobals.lookup_params('population')
        seed = ExaGlobals.lookup_params('seed')
        
        random.seed(seed)
        rng = np.random.default_rng(seed=seed)

        for p in range(population):
            G = igraph.Graph.K_Regular(nodes, regularity)
            while not G.is_connected():
                clusters = G.clusters().subgraphs()
                remove = []
                add = [] 
                R = rng.shuffle(range(len(clusters)))
                for (a,b) in zip(R[::2],R[1::2]):
                    ea = rng.choice(clusters[a].get_edgelist())
                    eb = rng.choice(clusters[b].get_edgelist())
                    remove.extend([ea,eb])
                    if rng.random() < .5:
                        add.extend(zip(ea,eb))
                    else:
                        add.extend(zip(ea,reversed(eb)))
                G.remove_edges(remove)
                G.add_edges(add)

            with open(self.getFileName(0, p),'w') as output:
                for e in G.get_edgelist():
                    (u,v) = (min(e),max(e))
                    output.write(repr(u) + " " + repr(v) + "\n")

        # Create spaces
        float_max = np.finfo(np.float64).max
        self.low = low = np.array([0, 0], dtype=np.float64)
        high = np.array([float_max, float_max], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = gym.spaces.Discrete(1)
        
        self.command = (
            'mpirun -n 1 ' +
            '/people/firo017/sst_macro/sst-macro/build/install/bin/sstmac ' + 
            '-f /people/firo017/sst_macro/sst-macro/build/random/parameters_offered_load_random_lps_8k_2p00ms.ini ' +
            '-p topology.name=file ' + 
            '-p topology.filename=/people/firo017/sst_macro/sst-macro/build/test_graph.json ' +
            '-p topology.routing_tables=/people/firo017/sst_macro/sst-macro/build/test_graph_routes.json' 
            )
        
        # Init states
        self.initial_state = low
        self.state = low

    def getFileName(self, gen, num):
        name = "_".join(["graph", str(gen), str(ExaComm.global_comm.rank), str(num), ".txt"])
        return os.path.join(self.dir, name)

    def getCommand(self, filename, r):
        command = (
            'mpirun -n 1 ' +
            '/people/firo017/sst_macro/sst-macro/build/install/bin/sstmac ' + 
            '-f /people/firo017/sst_macro/sst-macro/build/random/parameters_offered_load_random_lps_8k_2p00ms.ini ' +
            '-p topology.name=file ' + 
            '-p topology.filename=/people/firo017/sst_macro/sst-macro/build/test_graph.json ' +
            '-p topology.routing_tables=/people/firo017/sst_macro/sst-macro/build/test_graph_routes.json' 
            )

    def step(self, action):
        # Run Command
        stream = os.popen(self.command)
        output = stream.read()
        lines = output.split("\n")
        
        # # Process results
        # runtime_line = [line for line in lines if "Estimated total runtime of" in line]
        # runtime = [np.float64(x) for x in runtime_line[0].split() if "." in x]
        # exp_time_line = [line for line in lines if "ST/macro ran for" in line]
        # exp_time = [np.float64(x) for x in exp_time_line[0].split() if "." in x]

        # Generate the next state
        # next_state = np.array([runtime[0], exp_time[0]], dtype=np.float64)
        # print(next_state)
        # return next_state, 1, True, {}
        return self.low, 1, True, {}

    def reset(self):
        self.state = self.initial_state
        return self.state

    def jsonify(self, filename, concentration):
        with open(filename,'r') as input:
            E = [list(map(int,line.split())) for line in input]

        n = 1 + max([u for (u,v) in E] + [v for (u,v) in E])

        # Build graph structure
        # graph[v] = adjacency_list
        # position in list determines port of connection
        graph = {v : [] for v in range(n)}
        for (u,v) in E:
            graph[u].append(v)
            graph[v].append(u)
        for v in range(n):
            graph[v].sort()

        previous_port = {v : 0 for v in range(n)}
        topology = { "avg_num_hops" : -1, "switches": OrderedDict(), "nodes" : OrderedDict()}
        for v in range(n):
            outports = [(len(graph[v]) +i , {"destination" : "node" + repr(v * concentration + i), "inport" : 0}) for i in range(concentration)]
            for j in range(concentration):
                topology["nodes"]["node" + repr(v * concentration + j)] = {"outports" : {"0" : {"destination" : "switch" + repr(v), "inport" : len(graph[v]) + j}}}
            for (i,u) in enumerate(graph[v]):
                outports.append((i,{"destination": "switch"+repr(u), "inport" : previous_port[u]}))
                previous_port[u]+= 1
            outports.sort()
            topology["switches"]["switch"+repr(v)] = {"outports" : OrderedDict(outports)}

        with open( filename[:-4] + ".json", 'w') as output:
            json.dump(topology, output, indent=2)

        # Switch graph format so we can look up outports
        for v in range(n):
            graph[v] = { u : i for (i,u) in enumerate(graph[v])}

        route_json = {"switches" : OrderedDict([("switch" + repr(i),{"routes": OrderedDict()}) for i in range(n)])}   

        G = igraph.Graph()
        G.add_vertices(range(n))
        G.add_edges(E)

        for v in range(n):
            current_switch = "switch" + repr(v)
            r = 0
            paths = G.get_all_shortest_paths(v)
            #print(paths)
            routes = { u : set([]) for u in range(n)}
            hopcount = {u : -1 for u in range(n)}
            for P in paths:
                hopcount[P[-1]] = len(P)
                if len(P) == 1:
                    # Starting port for nodes
                    routes[v] = len(graph[v])
                else:
                    routes[P[-1]].add(graph[v][P[1]])
            for u in range(n):
                if u == v:
                    for c in range(concentration):
                        route_json["switches"][current_switch]["routes"]["route" + repr(r)] = {"node" : concentration * u + c, "outport" : routes[v] + c, "hopcount" : hopcount[u]}
                        r += 1
                else:
                    routes[u] = sorted(list(routes[u]))
                    for c in range(concentration):
                        for port in routes[u]:
                            route_json["switches"][current_switch]["routes"]["route" + repr(r)] = {"node" : concentration * u + c, "outport" : port, "hopcount" : hopcount[u]}
                            r += 1

        with open(filename[:-4] + "_routes.json",'w') as output:
            json.dump(route_json,output,indent=2)
                
            
        