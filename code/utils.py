import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

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
