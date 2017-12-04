from utils import comment_tokenizer
from nltk.probability import FreqDist
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from collections import Counter
import networkx as nx

from sklearn.metrics import roc_auc_score

class SentimentClassifier(object):

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.sorted_features = None
        self.number_dropped_features = 0

    def set_number_dropped_features(self,number_dropped_features):
        self.number_dropped_features = number_dropped_features
        print 'Set to drop the first top {} features ( based on MI ) from the comments'.format(number_dropped_features)

    def score(self,comments,votes):
        if isinstance(comments,list):
            vectors = self.vectorizer.transform([' '.join(comment_tokenizer(comment,dropped_features=self.sorted_features[:self.number_dropped_features])) for comment in comments])
            return self.model.score(vectors,votes)
        else:
            raise ValueError('Detected {} instead of list'.format(type(comments)))

    def predict(self,comments,number_dropped_features=0):
        if isinstance(comments,list):
            vectors = self.vectorizer.transform([' '.join(comment_tokenizer(comment,dropped_features=self.sorted_features[:self.number_dropped_features])) for comment in comments])
            return list(self.model.predict(vectors))
        else:
            raise ValueError('Detected {} instead of list'.format(type(comments)))

    def predict_proba(self,comments,number_dropped_features=0):
        if isinstance(comments,list):
            vectors = self.vectorizer.transform([' '.join(comment_tokenizer(comment,dropped_features=self.sorted_features[:self.number_dropped_features])) for comment in comments])
            return list(self.model.predict_proba(vectors))
        else:
            raise ValueError('Detected {} instead of list'.format(type(comments)))
    
    def loadWikipediaCorpusModel(self):
        if os.path.exists(os.path.join('..','models','sentimentClassifier.pkl')):
            print '--SENTIMENT CLASSIFIER LOADED--'
            sentimentClassifier = pickle.load(open(os.path.join('..','models','sentimentClassifier.pkl'),'rb'))
            self.vectorizer = sentimentClassifier.vectorizer
            self.model = sentimentClassifier.model
            self.sorted_features = sentimentClassifier.sorted_features
        else:
            print '--CREATION OF THE SENTIMENT CLASSIFIER--'
            print 'Collect and process comments and votes...'
            G = pickle.load(open(os.path.join('..','data','wikipedia','graph.pkl'),'rb'))
            edges = G.edges(data=True)
            comments = []
            votes = []
            for edge in edges:
                comments.append(edge[2]['text'])
                votes.append(edge[2]['vote'])
                        
            #Remove empty comments after processing so that we get non-empty comments in the sample
            size_comments = len(comments)
            for i in range(size_comments-1,-1,-1):
                if len(comment_tokenizer(comments[i]))==0:
                    del votes[i]
                    del comments[i]
            print 'Select 10000 most common words as features'
            #Select 10000 most common words as features
            fdist = FreqDist([word for comment in comments for word in comment_tokenizer(comment)])
            features = fdist.most_common(10000)
            features = [ feature[0] for feature in features ]

            print 'Sample comments with up and down vote...'
            #Sample 1000 comments with the vote
            index_sample_comments = np.random.choice(len(comments),1000,replace=False)
            sample_comments = []
            sample_votes = []
            for index in index_sample_comments:
                sample_comments.append(comments[index])
                sample_votes.append(votes[index])
            counter = Counter(sample_votes)
            print '... {} upvotes and {} downvotes for the sample'.format(counter[1],counter[-1])

            print 'Creation of the Logistic Regression model'
            #Do the logistic regression
            vectorizer = CountVectorizer(vocabulary=features)
            X = vectorizer.transform([ ' '.join(comment_tokenizer(comment)) for comment in sample_comments])
            y = sample_votes
                        
            logisticRegression = LogisticRegression(class_weight='balanced')
            logisticRegression.fit(X,y)
            print 'Precision of the model on the training set : {}'.format(logisticRegression.score(X,y))

            #Add Mutual Information to the word features
            print 'Save the features ranked by their MI'
            mutual_info = mutual_info_classif(X,y)
            sorted_features = [ feature for feature,_ in sorted(zip(features,mutual_info), key=lambda pair:pair[1],reverse=True)]

            self.sorted_features = sorted_features
            self.model = logisticRegression
            self.vectorizer = vectorizer
            pickle.dump(self,open(os.path.join('..','models','sentimentClassifier.pkl'),'wb'))
            print '--SENTIMENT CLASSIFIER BUILT AND SAVED IN MODELS--'

