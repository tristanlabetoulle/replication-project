from preprocessing import get_training_test_sets_wikipedia,get_training_test_sets_convote
from sentiment_analysis import SentimentClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from utils import wikipedia_sentiment_model_performance,wikipedia_random_model_performance,wikipedia_network_model_performance,wikipedia_network_sentiment_model_performance
from utils import convote_sentiment_model_performance,convote_random_model_performance,convote_network_model_performance,convote_network_sentiment_model_performance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as ticker

gold_color = tuple([float(230)/255,float(153)/255,0])
blue_color = tuple([0,float(115)/255,float(179)/255])

def plot_wikipedia_auc_roc_evidence_ratio():
    evidence_ratios = [0.125,0.25,0.5,0.75]
    df = pd.DataFrame(columns=['Evidence ratio','Model','Area under the curve'])
    for evidence_ratio in evidence_ratios:
        for i in range(10):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_wikipedia(i,evidence_ratio=evidence_ratio)
            
            y,y_pred = wikipedia_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Random',roc_auc_score(y,y_pred)]
            print 'Score Random : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = wikipedia_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Sentiment',roc_auc_score(y,y_pred)]
            print 'Score Sentiment : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = wikipedia_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Network',roc_auc_score(y,y_pred)]
            print 'Score Network : {}'.format(roc_auc_score(y,y_pred))
            
            y,y_pred = wikipedia_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Combined',roc_auc_score(y,y_pred)]
            print 'Score Combined : {}'.format(roc_auc_score(y,y_pred))

    sns.pointplot(x='Evidence ratio',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','wikipedia_AUC_ROC_evidence_ratio.png'))
    plt.clf()

def plot_wikipedia_auc_negPR_evidence_ratio():
    evidence_ratios = [0.125,0.25,0.5,0.75]
    df = pd.DataFrame(columns=['Evidence ratio','Model','Area under the curve'])
    for evidence_ratio in evidence_ratios:
        for i in range(10):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_wikipedia(i,evidence_ratio=evidence_ratio)
            
            y,y_pred = wikipedia_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Random',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Random : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Sentiment',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Sentiment : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Network',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Network : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Combined',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Combined : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

    sns.pointplot(x='Evidence ratio',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','wikipedia_AUC_negPR_evidence_ratio.png'))
    plt.clf()

def plot_wikipedia_auc_roc_number_dropped_features():
    number_dropped_features_plot = [0,10,50,100,500,1000,2000]
    df = pd.DataFrame(columns=['Number of dropped features','Model','Area under the curve'])
    for number_dropped_features in number_dropped_features_plot:
        for i in range(10):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_wikipedia(i,evidence_ratio=0.75)
            
            y,y_pred = wikipedia_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[number_dropped_features,'Random',roc_auc_score(y,y_pred)]
            print 'Score Random : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = wikipedia_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=number_dropped_features)
            df.loc[len(df)]=[number_dropped_features,'Sentiment',roc_auc_score(y,y_pred)]
            print 'Score Sentiment : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = wikipedia_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[number_dropped_features,'Network',roc_auc_score(y,y_pred)]
            print 'Score Network : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = wikipedia_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=number_dropped_features)
            df.loc[len(df)]=[number_dropped_features,'Combined',roc_auc_score(y,y_pred)]
            print 'Score Combined : {}'.format(roc_auc_score(y,y_pred))

    sns.pointplot(x='Number of dropped features',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','wikipedia_AUC_ROC_number_dropped_features.png'))
    plt.clf()

def plot_wikipedia_auc_negPR_number_dropped_features():
    number_dropped_features_plot = [0,10,50,100,500,1000,2000]
    df = pd.DataFrame(columns=['Number of dropped features','Model','Area under the curve'])
    for number_dropped_features in number_dropped_features_plot:
        for i in range(10):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_wikipedia(i,evidence_ratio=0.75)
            
            y,y_pred = wikipedia_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[number_dropped_features,'Random',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Random : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=number_dropped_features)
            df.loc[len(df)]=[number_dropped_features,'Sentiment',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Sentiment : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[number_dropped_features,'Network',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Network : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

            y,y_pred = wikipedia_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=number_dropped_features)
            df.loc[len(df)]=[number_dropped_features,'Combined',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
            print 'Score Combined : {}'.format(average_precision_score([1-i for i in y],[1-i for i in y_pred]))

    sns.pointplot(x='Number of dropped features',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','wikipedia_AUC_negPR_number_dropped_features.png'))
    plt.clf()

def plot_convote_auc_roc_evidence_ratio():
    evidence_ratios = [0.05,0.1,0.125,0.15,0.2,0.25]
    df = pd.DataFrame(columns=['Evidence ratio','Model','Area under the curve'])
    for evidence_ratio in evidence_ratios:
        for i in range(5):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(i,evidence_ratio=evidence_ratio)

            y,y_pred = convote_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Random',roc_auc_score(y,y_pred)]
            print 'Score Random : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = convote_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Sentiment',roc_auc_score(y,y_pred)]
            print 'Score Sentiment : {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = convote_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Network',roc_auc_score(y,y_pred)]
            print 'Score Network: {}'.format(roc_auc_score(y,y_pred))

            y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Combined',roc_auc_score(y,y_pred)]
            print 'Score Combined : {}'.format(roc_auc_score(y,y_pred))
        
    sns.pointplot(x='Evidence ratio',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','convote_AUC_ROC_evidence_ratio.png'))
    plt.clf()

def plot_convote_auc_negPR_evidence_ratio():
    evidence_ratios = [0.05,0.1,0.125,0.15,0.2,0.25]
    df = pd.DataFrame(columns=['Evidence ratio','Model','Area under the curve'])
    for evidence_ratio in evidence_ratios:
        for i in range(5):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(i,evidence_ratio=evidence_ratio)

            y,y_pred = convote_random_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Random',average_precision_score([1-i for i in y],[1-i for i in y_pred])]

            y,y_pred = convote_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Sentiment',average_precision_score([1-i for i in y],[1-i for i in y_pred])]

            y,y_pred = convote_network_model_performance(testing_set_graph,full_testing_set_graph)
            df.loc[len(df)]=[evidence_ratio,'Network',average_precision_score([1-i for i in y],[1-i for i in y_pred])]

            y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            df.loc[len(df)]=[evidence_ratio,'Combined',average_precision_score([1-i for i in y],[1-i for i in y_pred])]
        
    sns.pointplot(x='Evidence ratio',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,'k'],linestyles=['--','-','-','-'],markers=['o','o','o','o'],ci=95)
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.savefig(os.path.join('..','results','convote_AUC_negPR_evidence_ratio.png'))
    plt.clf()


def plot_convote_positive_precision_recall():
    training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(0,evidence_ratio=0.15)

    y,y_pred = convote_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    plt.plot(recall,precision,color=gold_color,label='Sentiment',linewidth=2)
  
    y,y_pred = convote_network_model_performance(testing_set_graph,full_testing_set_graph)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    plt.plot(recall,precision,color=blue_color,label='Network',linewidth=2)
    
    y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    plt.plot(recall,precision,color='k',label='Combined',linewidth=2)

    plt.legend(loc='lower right')
    plt.ylim(ymin=0)
    plt.xlabel('Positive recall')
    plt.ylabel('Positive precision')
    plt.savefig(os.path.join('..','results','convote_positive_precision_recall.png'))
    plt.clf()
    
def plot_convote_negative_precision_recall():
    training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(0,evidence_ratio=0.15)

    y,y_pred = convote_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
    precision, recall, _ = precision_recall_curve([1-i for i in y],[1-i for i in y_pred])
    plt.plot(recall,precision,color=gold_color,label='Sentiment',linewidth=2)
        
    y,y_pred = convote_network_model_performance(testing_set_graph,full_testing_set_graph)
    precision, recall, _ = precision_recall_curve([1-i for i in y],[1-i for i in y_pred])
    plt.plot(recall,precision,color=blue_color,label='Network',linewidth=2)
    
    y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
    precision, recall, _ = precision_recall_curve([1-i for i in y],[1-i for i in y_pred])
    plt.plot(recall,precision,color='k',label='Combined',linewidth=2)

    plt.legend(loc='lower right')
    plt.ylim(ymin=0)
    plt.xlabel('Negative recall')
    plt.ylabel('Negative precision')
    plt.savefig(os.path.join('..','results','convote_negative_precision_recall.png'))
    plt.clf()
    
#plot_wikipedia_auc_roc_evidence_ratio()
#plot_wikipedia_auc_negPR_evidence_ratio()
#plot_wikipedia_auc_roc_number_dropped_features()
#plot_wikipedia_auc_negPR_number_dropped_features()
#plot_convote_auc_roc_evidence_ratio()
#plot_convote_auc_negPR_evidence_ratio()
#plot_convote_positive_precision_recall()
#plot_convote_negative_precision_recall()
