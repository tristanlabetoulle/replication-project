import fileinput
import os
import re
import sys
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
from scipy.stats import linregress
import pickle
import scipy.stats

def change_weight_prior_psl_convote(weight_prior):
    for count,line in enumerate(fileinput.input(os.path.join('..','cli','convote_network_sentiment_model.psl'),inplace=True)):
        exp = str(weight_prior)+'\g<2>'
        if count==0 or count==1:
             line= re.sub(r'(.+)(:)',exp,line)
        sys.stdout.write(line)
    fileinput.close()

def extension_plot_convote_auc_roc_evidence_ratio():
    gold_color = tuple([float(230)/255,float(153)/255,0])
    blue_color = tuple([0,float(115)/255,float(179)/255])
    
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

    prior_weights = [0.001,0.003,1.0,100.0,1000.0]
    black_colors = [tuple([i*float(60)/255,i*float(60)/255,i*float(60)/255]) for i in range(len(prior_weights))]
    for prior_weight in [0.001,0.003,1.0,100.0,1000.0]:
        change_weight_prior_psl_convote(prior_weight)
        for evidence_ratio in evidence_ratios:
            for i in range(5):
                training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(i,evidence_ratio=evidence_ratio)
                
                y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
                df.loc[len(df)]=[evidence_ratio,'Combined PW '+str(prior_weight),roc_auc_score(y,y_pred)]
                print 'Score Combined for prior weight of {} and evidence ratio of {} : {}'.format(prior_weight,evidence_ratio,roc_auc_score(y,y_pred))
    change_weight_prior_psl_convote(1.0)
    g = sns.pointplot(x='Evidence ratio',y='Area under the curve',hue='Model',data=df,palette=['r',gold_color,blue_color,black_colors[0],black_colors[1],black_colors[2],black_colors[3],black_colors[4]],linestyles=['--','-','-','-','-','-','-','-'],markers=['.','.','.','.','.','.','.','.'],ci=95)
    plt.setp(g.lines,linewidth=2.0)
    plt.setp(g.collections, sizes=[50])
    plt.legend(loc='upper left')
    plt.ylim(ymax=1)
    plt.title('Variants of the prior weight (PW) for the Combined Model')
    plt.savefig(os.path.join('..','results','convote_AUC_ROC_evidence_ratio_test.png'))
    plt.clf()

def logistic_test_hypothesis():
    prior_weights = [0,0.001,0.002,0.0025,0.00275,0.003,0.004,0.005,0.01,0.015,0.1,0.15,1.0,1.5,10.0,15.0]
    scores = []
    for prior_weight in prior_weights:
        change_weight_prior_psl_convote(prior_weight)
        temp_scores = []
        for i in range(5):
            training_set_graph, testing_set_graph, full_training_set_graph, full_testing_set_graph = get_training_test_sets_convote(i,evidence_ratio=0.05)
            y,y_pred = convote_network_sentiment_model_performance(testing_set_graph,full_testing_set_graph,number_dropped_features=0)
            temp_scores.append(roc_auc_score(y,y_pred))
            print 'Score Combined for prior weight of {} and evidence ratio of 0.05 : {}'.format(prior_weight,0.05,roc_auc_score(y,y_pred))
        scores.append(np.mean(temp_scores))
    change_weight_prior_psl_convote(1.0)
    plt.plot(prior_weights,scores,marker='o',markersize=3)
    plt.xlabel('Prior Weight (PW)')
    plt.ylabel('AUROC')
    plt.title('AUROC score against PW')
    plt.savefig(os.path.join('..','results','AUROC_PW.png'))

#extension_plot_convote_auc_roc_evidence_ratio()
#logistic_test_hypothesis()
