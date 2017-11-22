import networkx as nx
import os
import re
import numpy as np
import pickle

def wikipedia_graph():
    G = nx.Graph()

    print '--CONSTRUCTION OF THE WIKIPEDIA GRAPH--'
    with open(os.path.join('..','data','wikipedia','rfa_all.txt'),'rb') as infile:
        node_data = []
        for line in infile:
            if line[0:3]=='SRC':
                node_data.append(line[4:-1])
            elif line[0:3]=='TGT':
                node_data.append(line[4:-1])
            elif line[0:3]=='VOT':
                node_data.append(int(line[4:-1]))
            elif line[0:3]=='TXT':
                node_data.append(line[4:-1])
            elif line=='\n':
                if node_data[2]!=0:
                    G.add_edge(node_data[0],node_data[1],vote=node_data[2],text=node_data[3])
                node_data = []

    nx.write_gpickle(G,os.path.join('..','data','wikipedia','graph.pkl'))
    print '--GRAPH STORED IN DATA--'

def wikipedia_sets(number_sets=10,number_nodes=350):
    print '--CREATION OF THE GRAPH SETS--'
    G = nx.read_gpickle(os.path.join('..','data','wikipedia','graph.pkl'))
    random_indexes = np.random.randint(0,G.number_of_nodes(),number_sets)
    nodes_name = list(G.nodes())
    subgraphs_set = []
    for index in random_indexes:
        queue = []
        queue.append(nodes_name[index])
        list_nodes = []
        while len(list_nodes)<=number_nodes:
            node_name = queue.pop(0)
            neighbors = G[node_name].keys()
            neighbors = [ neighbor for neighbor in neighbors if neighbor not in list_nodes]
            for neighbor in neighbors :
                list_nodes.append(neighbor)
                queue.append(neighbor)
        subgraphs_set.append(nx.Graph(G.subgraph(list_nodes[:350])))
    pickle.dump(subgraphs_set,open(os.path.join('..','data','wikipedia','subgraphs_set.pkl'),'wb'))
    print '--{} GRAPH SETS CREATED WITH {} NODES EACH--'.format(number_sets,number_nodes)

def get_training_test_sets(training_set_subgraph,evidence_ratio=1.0):
    print '--CREATING THE TRAINING AND TESTING SETS--'
    subgraphs_set = pickle.load(open(os.path.join('..','data','wikipedia','subgraphs_set.pkl'),'rb'))
    full_testing_set_subgraph = (training_set_subgraph+1)%len(subgraphs_set)
    full_training_set_subgraph = subgraphs_set[training_set_subgraph]
    full_testing_set_subgraph = subgraphs_set[full_testing_set_subgraph]
    removed_edges = [ edge for edge in full_testing_set_subgraph.edges() if edge in full_training_set_subgraph.edges()]
    full_testing_set_subgraph.remove_edges_from(removed_edges)

    training_set_subgraph = full_training_set_subgraph.copy()
    testing_set_subgraph = full_testing_set_subgraph.copy()
    #remove votes for evidence_ratio
    print 'Keep {}% of the original evidence ( votes ) and change the others to \'None\'...'.format(evidence_ratio*100)
    indexes_remove_votes_training = np.random.choice(training_set_subgraph.number_of_edges(),int(training_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace='False')
    for count,edge in enumerate(training_set_subgraph.edges()):
        if count in indexes_remove_votes_training:
            training_set_subgraph[edge[0]][edge[1]]['vote']=None
    indexes_remove_votes_testing = np.random.choice(testing_set_subgraph.number_of_edges(),int(testing_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace='False')
    for count,edge in enumerate(testing_set_subgraph.edges()):
        if count in indexes_remove_votes_testing:
            testing_set_subgraph[edge[0]][edge[1]]['vote']=None
    
    print 'Training Edges : {}'.format(training_set_subgraph.number_of_edges())
    print 'Testing Edges : {}'.format(testing_set_subgraph.number_of_edges())
    print '--REMOVED {} OVERLAPPING EDGES--'.format(len(removed_edges))
    return training_set_subgraph, testing_set_subgraph, full_training_set_subgraph, full_testing_set_subgraph

#1) Create the graph
#wikipedia_graph()
#2) Create the sets
#wikipedia_sets()
#3) Get the training and testing sets with non-overlapping edges
#get_training_test_sets(0)
            
        
