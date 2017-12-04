import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
import os
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import csv
import subprocess

def comment_tokenizer(text,dropped_features=[]):

    BLACKLIST_STOPWORDS = ['over','only','very','not','no']
    ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
    NEG_CONTRACTIONS = [
        (r'aren\'t', 'are not'),
        (r'can\'t', 'can not'),
        (r'couldn\'t', 'could not'),
        (r'daren\'t', 'dare not'),
        (r'didn\'t', 'did not'),
        (r'doesn\'t', 'does not'),
        (r'don\'t', 'do not'),
        (r'isn\'t', 'is not'),
        (r'hasn\'t', 'has not'),
        (r'haven\'t', 'have not'),
        (r'hadn\'t', 'had not'),
        (r'mayn\'t', 'may not'),
        (r'mightn\'t', 'might not'),
        (r'mustn\'t', 'must not'),
        (r'needn\'t', 'need not'),
        (r'oughtn\'t', 'ought not'),
        (r'shan\'t', 'shall not'),
        (r'shouldn\'t', 'should not'),
        (r'wasn\'t', 'was not'),
        (r'weren\'t', 'were not'),
        (r'won\'t', 'will not'),
        (r'wouldn\'t', 'would not'),
        (r'ain\'t', 'am not') # not only but stopword anyway
    ]
    OTHER_CONTRACTIONS = {
        "'m": 'am',
        "'ll": 'will',
        "'s": 'has', # or 'is' but both are stopwords
        "'d": 'had'  # or 'would' but both are stopwords
    }

    # decode in utf-8
    text = text.decode('utf-8')
    # lowercase
    doc = text.lower()
    # transform negative contractions (e.g don't --> do not)
    for t in NEG_CONTRACTIONS:
        doc = re.sub(t[0], t[1], doc)
    # tokenize
    tokens = nltk.word_tokenize(doc)
    # transform other contractions (e.g 'll --> will)
    tokens = [OTHER_CONTRACTIONS[token] if OTHER_CONTRACTIONS.get(token) else token for token in tokens]
    # remove punctuation
    r = r'[a-z]+'
    tokens = [word for word in tokens if re.search(r, word)]
    # remove irrelevant stop words
    tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    # stemming
    tokens = [PorterStemmer().stem(token) for token in tokens]
    #filter out words with 'support' and 'oppos'
    tokens = [token for token in tokens if token!='support' and token!='oppos']

    #filter out dropped features
    tokens = [token for token in tokens if token not in dropped_features]

    return tokens

class PlattScaling(object):

    def __init__(self):
        self.lr = None

    def fit_convote(self):
        with open(os.path.join('..','data','convote','edges_individual_document.v1.1.csv'),'rb') as infile:
            SVM_outputs = []
            labels = []
            for line in infile:
                line = line.split(',')
                _,_,_,vote = line[0].split('_')
                if 'N' in vote:
                    vote = False
                else :
                    vote = True
                SVM_outputs.append(float(line[2]))
                labels.append(vote)
            lr = LogisticRegression()
            lr.fit(np.reshape(SVM_outputs,(-1,1)),labels)
            self.lr = lr

    def predict_proba(self,X):
        if isinstance(X,list):
            return self.lr.predict_proba(np.reshape(X,(1,-1)))
        else:
            raise ValueError('Detected {} instead of list'.format(type(X)))

def wikipedia_random_model_performance(graph,full_graph):
    X = []
    y = []
    for edge in graph.edges(data=True):
        if edge[2]['vote']==None:
            y.append(full_graph[edge[0]][edge[1]]['vote'])
    y_pred = np.random.rand(len(y))
    return y,y_pred

def wikipedia_sentiment_model_performance(graph,full_graph,number_dropped_features=0):
    from sentiment_analysis import SentimentClassifier
    sentimentClassifier = SentimentClassifier()
    sentimentClassifier.loadWikipediaCorpusModel()
    sentimentClassifier.set_number_dropped_features(number_dropped_features)
    X = []
    y = []
    temp=None
    for edge in graph.edges(data=True):
        if edge[2]['vote']==None:
            X.append(edge[2]['text'])
            y.append(full_graph[edge[0]][edge[1]]['vote'])
            temp=edge
    y_pred = sentimentClassifier.predict_proba(X)
    y_pred = [ pred[1] for pred in y_pred ]
    print ''
    return y,y_pred

def wikipedia_network_model_performance(graph,full_graph):
    testing_edges = {}
    with open(os.path.join('..','cli','wikipedia_knows_obs.txt'),'wb') as knows_obs:
        with open(os.path.join('..','cli','wikipedia_trusts_obs.txt'),'wb') as trusts_obs:
            with open(os.path.join('..','cli','wikipedia_trusts_targets.txt'),'wb') as trusts_targets:
                writer_knows_obs = csv.writer(knows_obs,delimiter='\t')
                writer_trusts_obs = csv.writer(trusts_obs,delimiter='\t')
                writer_trusts_targets = csv.writer(trusts_targets,delimiter='\t')
                for edge in list(graph.edges(data=True)):
                    writer_knows_obs.writerow([edge[0],edge[1]])
                    if edge[2]['vote']!=None:
                        writer_trusts_obs.writerow([edge[0],edge[1],edge[2]['vote']])
                    else:
                        writer_trusts_targets.writerow([edge[0],edge[1]])
                        testing_edges[tuple([edge[0],edge[1]])]=full_graph[edge[0]][edge[1]]['vote']
    if os.path.exists(os.path.join('..','cli','wikipedia_results.txt')):
        os.remove(os.path.join('..','cli','wikipedia_results.txt'))
    if os.path.exists('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db'):
        os.remove('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db')
    process = subprocess.Popen('cd .. & cd cli & java -jar psl-cli-2.0.0.jar -infer -model wikipedia_network_model.psl -data wikipedia_network_model.data >> wikipedia_results.txt',shell=True)
    process.wait()

    y = []
    y_pred = []
    count_lines_for_error = 0
    with open(os.path.join('..','cli','wikipedia_results.txt'),'rb') as infile:
        for line in infile:
            if line[:6]=='TRUSTS':
                count_lines_for_error = count_lines_for_error + 1
                edge = re.search(r'(?:\()(.+)(?:\))',line).group(1)
                edge = edge.split('\', \'')
                edge = [edge[0][1:],edge[1][:-1]]
                vote = float(re.search(r'(?:\=)(.+)',line).group(1))
                if tuple(edge) in testing_edges:
                    y.append(testing_edges[tuple(edge)])
                    y_pred.append(vote)
    print len(y),len(y_pred)
    if count_lines_for_error==0:
        raise ValueError('Error in the CLI system, make sure the evidence ratio is not 1.0')
    return y,y_pred

def wikipedia_network_sentiment_model_performance(graph,full_graph,number_dropped_features=0):
    from sentiment_analysis import SentimentClassifier
    sentimentClassifier = SentimentClassifier()
    sentimentClassifier.loadWikipediaCorpusModel()
    sentimentClassifier.set_number_dropped_features(number_dropped_features)

    #print list(graph.edges(data=True))[-1]
    testing_edges = {}
    with open(os.path.join('..','cli','wikipedia_prior_obs.txt'),'wb') as prior_obs:
        with open(os.path.join('..','cli','wikipedia_knows_obs.txt'),'wb') as knows_obs:
            with open(os.path.join('..','cli','wikipedia_trusts_obs.txt'),'wb') as trusts_obs:
                with open(os.path.join('..','cli','wikipedia_trusts_targets.txt'),'wb') as trusts_targets:
                    writer_prior_obs = csv.writer(prior_obs,delimiter='\t')
                    writer_knows_obs = csv.writer(knows_obs,delimiter='\t')
                    writer_trusts_obs = csv.writer(trusts_obs,delimiter='\t')
                    writer_trusts_targets = csv.writer(trusts_targets,delimiter='\t')
                    for edge in list(graph.edges(data=True)):
                        writer_prior_obs.writerow([edge[0],edge[1],sentimentClassifier.predict_proba([edge[2]['text']])[0][1]])
                        writer_knows_obs.writerow([edge[0],edge[1]])
                        if edge[2]['vote']!=None:
                            writer_trusts_obs.writerow([edge[0],edge[1],edge[2]['vote']])
                        else:
                            writer_trusts_targets.writerow([edge[0],edge[1]])
                            testing_edges[tuple([edge[0],edge[1]])]=full_graph[edge[0]][edge[1]]['vote']
    if os.path.exists(os.path.join('..','cli','wikipedia_results.txt')):
        os.remove(os.path.join('..','cli','wikipedia_results.txt'))
    if os.path.exists('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db'):
        os.remove('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db')
    process = subprocess.Popen('cd .. & cd cli & java -jar psl-cli-2.0.0.jar -infer -model wikipedia_network_sentiment_model.psl -data wikipedia_network_sentiment_model.data >> wikipedia_results.txt',shell=True)
    process.wait()

    y = []
    y_pred = []
    count_lines_for_error = 0
    with open(os.path.join(os.getcwd(),'..','cli','wikipedia_results.txt'),'rb') as infile:
        for line in infile:
            if line[:6]=='TRUSTS':
                count_lines_for_error = count_lines_for_error + 1
                edge = re.search(r'(?:\()(.+)(?:\))',line).group(1)
                edge = edge.split('\', \'')
                edge = [edge[0][1:],edge[1][:-1]]
                vote = float(re.search(r'(?:\=)(.+)',line).group(1))
                if tuple(edge) in testing_edges:
                    y.append(testing_edges[tuple(edge)])
                    y_pred.append(vote)
    if count_lines_for_error==0:
        raise ValueError('Error in the CLI system, make sure the evidence ratio is not 1.0')
    return y,y_pred

def convote_random_model_performance(graph,full_graph):
    X = []
    y = []
    for edge in graph.edges(data=True):
        if edge[2]['vote_agree']==None:
            y.append(full_graph[edge[0]][edge[1]]['vote_agree'])
    y_pred = np.random.rand(len(y))
    return y,y_pred

def convote_sentiment_model_performance(graph,full_graph,number_dropped_features=0):
    X = []
    y = []
    y_pred = []
    for edge in graph.edges(data=True):
        if edge[2]['vote_agree']==None:
            y_pred.append(graph[edge[0]][edge[1]]['sentiment_agree'])
            y.append(full_graph[edge[0]][edge[1]]['vote_agree'])
    return y,y_pred

def convote_network_model_performance(graph,full_graph):
    testing_edges = {}
    with open(os.path.join('..','cli','convote_covoted_obs.txt'),'wb') as covoted_obs:
        with open(os.path.join('..','cli','convote_agree_obs.txt'),'wb') as agree_obs:
            with open(os.path.join('..','cli','convote_agree_targets.txt'),'wb') as agree_targets:
                writer_covoted_obs = csv.writer(covoted_obs,delimiter='\t')
                writer_agree_obs = csv.writer(agree_obs,delimiter='\t')
                writer_agree_targets = csv.writer(agree_targets,delimiter='\t')
                for edge in list(graph.edges(data=True)):
                    writer_covoted_obs.writerow([edge[0],edge[1]])
                    if edge[2]['vote_agree']!=None:
                        writer_agree_obs.writerow([edge[0],edge[1],edge[2]['vote_agree']])
                    else:
                        writer_agree_targets.writerow([edge[0],edge[1]])
                        testing_edges[tuple([edge[0],edge[1]])]=full_graph[edge[0]][edge[1]]['vote_agree']
    if os.path.exists(os.path.join('..','cli','convote_results.txt')):
        os.remove(os.path.join('..','cli','convote_results.txt'))
    if os.path.exists('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db'):
        os.remove('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db')
    process = subprocess.Popen('cd .. & cd cli & java -jar psl-cli-2.0.0.jar -infer -model convote_network_model.psl -data convote_network_model.data >> convote_results.txt',shell=True)
    process.wait()

    y = []
    y_pred = []
    count_lines_for_error = 0
    with open(os.path.join('..','cli','convote_results.txt'),'rb') as infile:
        for line in infile:
            if line[:5]=='AGREE':
                count_lines_for_error = count_lines_for_error + 1
                edge = re.search(r'(?:\()(.+)(?:\))',line).group(1)
                edge = edge.split('\', \'')
                edge = [edge[0][1:],edge[1][:-1]]
                vote = float(re.search(r'(?:\=)(.+)',line).group(1))
                if tuple(edge) in testing_edges:
                    y.append(testing_edges[tuple(edge)])
                    y_pred.append(vote)
    if count_lines_for_error==0:
        raise ValueError('Error in the CLI system, make sure the evidence ratio is not 1.0')
    return y,y_pred

def convote_network_sentiment_model_performance(graph,full_graph,number_dropped_features=0):
    testing_edges = {}
    with open(os.path.join('..','cli','convote_prior_obs.txt'),'wb') as prior_obs:
        with open(os.path.join('..','cli','convote_covoted_obs.txt'),'wb') as covoted_obs:
            with open(os.path.join('..','cli','convote_agree_obs.txt'),'wb') as agree_obs:
                with open(os.path.join('..','cli','convote_agree_targets.txt'),'wb') as agree_targets:
                    writer_prior_obs = csv.writer(prior_obs,delimiter='\t')
                    writer_covoted_obs = csv.writer(covoted_obs,delimiter='\t')
                    writer_agree_obs = csv.writer(agree_obs,delimiter='\t')
                    writer_agree_targets = csv.writer(agree_targets,delimiter='\t')
                    for edge in list(graph.edges(data=True)):
                        writer_prior_obs.writerow([edge[0],edge[1],edge[2]['sentiment_agree']])
                        writer_covoted_obs.writerow([edge[0],edge[1]])
                        if edge[2]['vote_agree']!=None:
                            writer_agree_obs.writerow([edge[0],edge[1],edge[2]['vote_agree']])
                        else:
                            writer_agree_targets.writerow([edge[0],edge[1]])
                            testing_edges[tuple([edge[0],edge[1]])]=full_graph[edge[0]][edge[1]]['vote_agree']
    if os.path.exists(os.path.join('..','cli','convote_results.txt')):
        os.remove(os.path.join('..','cli','convote_results.txt'))
    if os.path.exists('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db'):
        os.remove('C:\\Users\\trist\\AppData\\Local\\Temp\\cli.mv.db')
    process = subprocess.Popen('cd .. & cd cli & java -jar psl-cli-2.0.0.jar -infer -model convote_network_sentiment_model.psl -data convote_network_sentiment_model.data >> convote_results.txt',shell=True)
    process.wait()

    y = []
    y_pred = []
    count_lines_for_error = 0
    with open(os.path.join(os.getcwd(),'..','cli','convote_results.txt'),'rb') as infile:
        for line in infile:
            if line[:5]=='AGREE':
                count_lines_for_error = count_lines_for_error + 1
                edge = re.search(r'(?:\()(.+)(?:\))',line).group(1)
                edge = edge.split('\', \'')
                edge = [edge[0][1:],edge[1][:-1]]
                vote = float(re.search(r'(?:\=)(.+)',line).group(1))
                if tuple(edge) in testing_edges:
                    y.append(testing_edges[tuple(edge)])
                    y_pred.append(vote)
    if count_lines_for_error==0:
        raise ValueError('Error in the CLI system, make sure the evidence ratio is not 1.0')
    return y,y_pred
