from preprocessing import get_training_test_sets_wikipedia,get_training_test_sets_convote
from sentiment_analysis import SentimentClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_wikipedia():
    sentimentClassifier = SentimentClassifier()
    sentimentClassifier.loadWikipediaCorpusModel()
    print ''
    statistics = {}

    roc_auc_random_model = []
    roc_auc_sentiment_model = []

    negPR_auc_random_model = []
    negPR_auc_sentiment_model = []

    for i in range(10):
        training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_wikipedia(i,evidence_ratio=0.5)
        testing_edges = testing_set_graph.edges(data=True)
        X = []
        y = []
        for edge in testing_edges:
            X.append(edge[2]['text'])
            y.append(full_testing_set_graph[edge[0]][edge[1]]['vote'])
        y_pred = sentimentClassifier.predict_proba(X,number_dropped_features = 0)
        y_pred = [ pred[1] for pred in y_pred ]
        print ''

        roc_auc_random_model.append(roc_auc_score(y,list(np.random.rand(len(y_pred)))))
        roc_auc_sentiment_model.append(roc_auc_score(y,y_pred))

        negPR_auc_random_model.append(average_precision_score([-i for i in y],[1-i for i in list(np.random.rand(len(y_pred)))]))
        negPR_auc_sentiment_model.append(average_precision_score([-i for i in y],[1-i for i in y_pred]))

    statistics['3a']={'sentiment':sum(roc_auc_sentiment_model)/len(roc_auc_sentiment_model),'random':sum(roc_auc_random_model)/len(roc_auc_random_model)}
    statistics['3b']={'sentiment':sum(negPR_auc_sentiment_model)/len(negPR_auc_sentiment_model),'random':sum(negPR_auc_random_model)/len(negPR_auc_random_model)}
    print statistics

def plot_convote():
    statistics = {}

    roc_auc_random_model = []
    roc_auc_sentiment_model = []

    negPR_auc_random_model = []
    negPR_auc_sentiment_model = []
    for i in range(5):
        training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(i,evidence_ratio=0.5)
        testing_edges = testing_set_graph.edges(data=True)
        X = []
        y = []
        y_pred=[]
        for edge in testing_edges:
            y_pred.append(edge[2]['sentiment_agree'])
            y.append(full_testing_set_graph[edge[0]][edge[1]]['vote_agree'])

        roc_auc_random_model.append(roc_auc_score(y,list(np.random.rand(len(y_pred)))))
        roc_auc_sentiment_model.append(roc_auc_score(y,y_pred))

        negPR_auc_random_model.append(average_precision_score([not i for i in y],[1-i for i in list(np.random.rand(len(y_pred)))]))
        negPR_auc_sentiment_model.append(average_precision_score([not i for i in y],[1-i for i in y_pred]))
        print ''

    statistics['7a']={'sentiment':sum(roc_auc_sentiment_model)/len(roc_auc_sentiment_model),'random':sum(roc_auc_random_model)/len(roc_auc_random_model)}
    statistics['7b']={'sentiment':sum(negPR_auc_sentiment_model)/len(negPR_auc_sentiment_model),'random':sum(negPR_auc_random_model)/len(negPR_auc_random_model)}
    print statistics

    precision,recall, _ = precision_recall_curve(y,y_pred)
    plt.step(recall,precision)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.show()

        
plot_convote()
#plot_wikipedia()
