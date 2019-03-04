#import numpy as np
import preprocessor as p
from nltk.tokenize import TweetTokenizer as tt
#from nltk.stem.porter import *
from string import punctuation
#from gensim.parsing.preprocessing import STOPWORDS
#import gensim
import pickle
from autocorrect import spell

def tokenize(tweet):
    tk=tt()
    #stemmer = PorterStemmer()
    p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.SMILEY)
    tweet=tweet.lower()
    tweet=p.clean(tweet)
    tweet=tweet.replace("'","")
    tokens=tk.tokenize(tweet)
    tokens=[w for w in tokens if w[0] not in punctuation]
    tokens = [spell(t) for t in tokens]
    #tokens = [stemmer.stem(t) for t in tokens]
    #tokens=[t for t in tokens if t not in STOPWORDS]
    
    
    return(tokens)

'''
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return(parsed_text)

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    stemmer = PorterStemmer()
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return(tokens)
'''

if __name__=="__main__":
    #t=np.load("tweets.npy")
    print(tokenize("What is aple???"))





