import networkx as nx
import os
import re
import numpy as np
import pickle
from utils import PlattScaling

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
    random_indexes = np.random.choice(range(G.number_of_nodes()),number_sets,replace=False)
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

def get_training_test_sets_wikipedia(training_set_subgraph,evidence_ratio=1.0):
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
    indexes_remove_votes_training = np.random.choice(training_set_subgraph.number_of_edges(),int(training_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace=False)
    number_removed_evidence_training = 0
    for count,edge in enumerate(training_set_subgraph.edges()):
        if count in indexes_remove_votes_training:
            training_set_subgraph[edge[0]][edge[1]]['vote']=None
            number_removed_evidence_training = number_removed_evidence_training + 1
    indexes_remove_votes_testing = np.random.choice(testing_set_subgraph.number_of_edges(),int(testing_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace=False)
    number_removed_evidence_testing = 0
    for count,edge in enumerate(testing_set_subgraph.edges()):
        if count in indexes_remove_votes_testing:
            testing_set_subgraph[edge[0]][edge[1]]['vote']=None
            number_removed_evidence_testing = number_removed_evidence_testing + 1
    
    print 'Training Edges : {}'.format(training_set_subgraph.number_of_edges())
    print 'Testing Edges : {}'.format(testing_set_subgraph.number_of_edges())
    print '--REMOVED {} OVERLAPPING EDGES--'.format(len(removed_edges))
    print '--REMOVED {} EVIDENCE IN TRAINING--'.format(number_removed_evidence_training)
    print '--REMOVED {} EVIDENCE IN TESTING--'.format(number_removed_evidence_testing)
    return training_set_subgraph, testing_set_subgraph, full_training_set_subgraph, full_testing_set_subgraph

#1) Create the graph
#wikipedia_graph()
#2) Create the sets
#wikipedia_sets()
#3) Get the training and testing sets with non-overlapping edges
#get_training_test_sets(0)
            
def convote_graph():

    print '--CONSTRUCTION OF THE CONVOTE GRAPH--'

    plattScaling = PlattScaling()
    plattScaling.fit_convote()
    print 'Platt Scaling initialized'

    with open(os.path.join('..','data','convote','edges_individual_document.v1.1.csv'),'rb') as infile:
        debate_to_speaker = {}
        for line in infile:
            line = line.split(',')
            debate,speaker,_,vote = line[0].split('_')
            if 'N' in vote:
                vote = False
            else :
                vote = True
            if not debate in debate_to_speaker:
                debate_to_speaker[debate] = {}
            if not speaker in debate_to_speaker[debate]:
                debate_to_speaker[debate][speaker]=[vote,[plattScaling.predict_proba([float(line[2])])[0][1]]]
            else:
                debate_to_speaker[debate][speaker][1].append(plattScaling.predict_proba([float(line[2])])[0][1])
    for debate in debate_to_speaker.keys():
        for speaker in debate_to_speaker[debate].keys():
            debate_to_speaker[debate][speaker][1]  = np.mean(debate_to_speaker[debate][speaker][1])
    #print debate_to_speaker['052']['400077']

    G = nx.Graph()
    for debate in debate_to_speaker.keys():
        speakers = list(debate_to_speaker[debate].keys())
        for i in range(len(speakers)):
            for j in range(i+1,len(speakers)):
                agree = debate_to_speaker[debate][speakers[i]][0]==debate_to_speaker[debate][speakers[j]][0]
                proba_agreement = debate_to_speaker[debate][speakers[i]][1]*debate_to_speaker[debate][speakers[j]][1]+(1-debate_to_speaker[debate][speakers[i]][1])*(1-debate_to_speaker[debate][speakers[j]][1])
                if G.has_edge(speakers[i],speakers[j]):
                    G[speakers[i]][speakers[j]]['vote_agree'].append(agree)
                    G[speakers[i]][speakers[j]]['sentiment_agree'].append(proba_agreement)
                else:
                    G.add_edge(speakers[i],speakers[j],vote_agree=[agree],sentiment_agree=[proba_agreement])

    positive_edges = 0
    for edge in G.edges():
        G[edge[0]][edge[1]]['vote_agree']=sum(G[edge[0]][edge[1]]['vote_agree'])>=(len(G[edge[0]][edge[1]]['vote_agree'])+1)/2
        G[edge[0]][edge[1]]['sentiment_agree']=np.mean(G[edge[0]][edge[1]]['sentiment_agree'])
        if G[edge[0]][edge[1]]['vote_agree']==True:
            positive_edges=positive_edges+1
    number_triangles = 0
    for node in G.nodes():
        number_triangles = number_triangles+nx.triangles(G,node)
    number_triangles = number_triangles/3
    print "Created the graph with {} edges ( {:.0f}% positive ) and {} triangles".format(G.number_of_edges(),float(positive_edges)/G.number_of_edges()*100,number_triangles)

    nx.write_gpickle(G,os.path.join('..','data','convote','graph.pkl'))
    print '--GRAPH STORED IN DATA--'

def convote_sets(number_sets=5):
    print '--CREATION OF THE GRAPH SETS--'
    G = nx.read_gpickle(os.path.join('..','data','convote','graph.pkl'))
    random_indexes = np.random.choice(range(G.number_of_edges()),(number_sets,G.number_of_edges()/number_sets),replace=False)
    edges_name = list(G.edges())
    subgraphs_set = []
    for i in range(len(random_indexes)):
        subgraph = G.copy()
        for j in range(len(edges_name)):
            if not j in random_indexes[i]:
                subgraph.remove_edge(*edges_name[j])
        subgraphs_set.append(subgraph)
    pickle.dump(subgraphs_set,open(os.path.join('..','data','convote','subgraphs_set.pkl'),'wb'))
    print '--{} GRAPH SETS CREATED WITH {} EDGES EACH--'.format(number_sets,G.number_of_edges()/number_sets)

def get_training_test_sets_convote(training_set_subgraph,evidence_ratio=1.0):
    print '--CREATING THE TRAINING AND TESTING SETS--'
    subgraphs_set = pickle.load(open(os.path.join('..','data','convote','subgraphs_set.pkl'),'rb'))
    full_testing_set_subgraph = (training_set_subgraph+1)%len(subgraphs_set)
    full_training_set_subgraph = subgraphs_set[training_set_subgraph]
    full_testing_set_subgraph = subgraphs_set[full_testing_set_subgraph]

    training_set_subgraph = full_training_set_subgraph.copy()
    testing_set_subgraph = full_testing_set_subgraph.copy()
    #remove votes for evidence_ratio
    print 'Keep {}% of the original evidence ( votes ) and change the others to \'None\'...'.format(evidence_ratio*100)
    indexes_remove_votes_training = np.random.choice(training_set_subgraph.number_of_edges(),int(training_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace=False)
    number_removed_evidence_training = 0
    for count,edge in enumerate(training_set_subgraph.edges()):
        if count in indexes_remove_votes_training:
            training_set_subgraph[edge[0]][edge[1]]['vote']=None
            number_removed_evidence_training = number_removed_evidence_training + 1
    indexes_remove_votes_testing = np.random.choice(testing_set_subgraph.number_of_edges(),int(testing_set_subgraph.number_of_edges()*(1-evidence_ratio)),replace=False)
    number_removed_evidence_testing = 0
    for count,edge in enumerate(testing_set_subgraph.edges()):
        if count in indexes_remove_votes_testing:
            testing_set_subgraph[edge[0]][edge[1]]['vote']=None
            number_removed_evidence_testing = number_removed_evidence_testing + 1
    
    print 'Training Edges : {}'.format(training_set_subgraph.number_of_edges())
    print 'Testing Edges : {}'.format(testing_set_subgraph.number_of_edges())
    print '--REMOVED {} EVIDENCE IN TRAINING--'.format(number_removed_evidence_training)
    print '--REMOVED {} EVIDENCE IN TESTING--'.format(number_removed_evidence_testing)
    return training_set_subgraph, testing_set_subgraph, full_training_set_subgraph, full_testing_set_subgraph

#1) Create the graph
#convote_graph()
#2) Create the sets
#convote_sets()
#3) Get the training and testing sets with non-overlapping edges
#get_training_test_sets_convote(0,evidence_ratio=0.6)
