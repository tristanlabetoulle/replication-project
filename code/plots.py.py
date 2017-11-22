from preprocessing import get_training_test_sets
from sentiment_analysis import SentimentClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
import networkx as nx
import numpy as np

def plot_auc_wikipedia_evidence_ratio():
    sentimentClassifier = SentimentClassifier()
    sentimentClassifier.loadWikipediaCorpusModel()
    print ''
    statistics = {}

    roc_auc_random_model = []
    roc_auc_sentiment_model = []

    negPR_auc_random_model = []
    negPR_auc_sentiment_model = []

    for i in range(10):
        training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets(i,evidence_ratio=0.5)
        testing_edges = testing_set_graph.edges(data=True)
        X = []
        y = []
        for edge in testing_edges:
            X.append(edge[2]['text'])
            y.append(full_testing_set_graph[edge[0]][edge[1]]['vote'])
        y_pred = sentimentClassifier.predict_proba(X,number_dropped_features = 1000)
        y_pred = [ pred[1] for pred in y_pred ]
        print ''

        roc_auc_random_model.append(roc_auc_score(y,list(np.random.rand(len(y_pred)))))
        roc_auc_sentiment_model.append(roc_auc_score(y,y_pred))

        negPR_auc_random_model.append(average_precision_score([-i for i in y],[1-i for i in list(np.random.rand(len(y_pred)))]))
        negPR_auc_sentiment_model.append(average_precision_score([-i for i in y],[1-i for i in y_pred]))

    statistics['3a']={'sentiment':sum(roc_auc_sentiment_model)/len(roc_auc_sentiment_model),'random':sum(roc_auc_random_model)/len(roc_auc_random_model)}
    statistics['3b']={'sentiment':sum(negPR_auc_sentiment_model)/len(negPR_auc_sentiment_model),'random':sum(negPR_auc_random_model)/len(negPR_auc_random_model)}
    print statistics
        
plot_auc_wikipedia_evidence_ratio()
