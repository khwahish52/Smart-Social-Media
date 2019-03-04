from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from string import punctuation
import preprocessor as p
from nltk.tokenize import TweetTokenizer as tt
from autocorrect import spell
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
#from flask_compress import Compress
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
#Compress(app)

'''
0=racist
1=sexist
2=hate
3=offensive
4=neutral
'''

'''
Dependencies:
*keras
*pickle
*numpy
*preprocessor  $ pip install tweet-preprocessor  https://github.com/s/preprocessor
*nltk
*autocorrect
'''

vocab_index_=[]
vocab_index=[]

# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model.h5")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# load json and create model
json_file = open('./model_rs.json', 'r')
loaded_model_json_ = json_file.read()
json_file.close()
loaded_model_ = model_from_json(loaded_model_json_)
# load weights into new model
loaded_model_.load_weights("./model_rs.h5")
# evaluate loaded model on test data
loaded_model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("success")

def sent_index(tweets):
    global vocab_index
    x=[]
    for t in tweets:
        tw=[]
        tokens=tokenize(t)
        for token in tokens:
            try:
                tw.append(vocab_index[token])
            except:
                pass
        x.append(tw)

    return(x)

def sent_index_(tweets):
    global vocab_index_
    x=[]
    for t in tweets:
        tw=[]
        tokens=tokenize(t)
        for token in tokens:
            try:
                tw.append(vocab_index_[token])
            except:
                pass
        x.append(tw)

    return(x)


def lstm(obj):
    global vocab_index
    global loaded_model
    f=open("./vocab_index_macy_.pickle","rb")
    vocab_index=pickle.load(f)
    tweets=np.array([obj])
    x=sent_index(tweets)
    x = pad_sequences(x, maxlen=49)
    
    y_pred=loaded_model.predict_on_batch(x)
    y_pred = np.argmax(y_pred, axis=1)
    return(y_pred[0])

def lstm_rs(obj):
    global vocab_index_
    global loaded_model_
    f=open("./vocab_index_new.pickle","rb")
    vocab_index_=pickle.load(f)
    tweets=np.array([obj])
    x=sent_index_(tweets)
    x = pad_sequences(x, maxlen=19)
    
    y_pred=loaded_model_.predict_on_batch(x)
    y_pred = np.argmax(y_pred, axis=1)
    return(y_pred[0])

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



@app.route('/predict', methods=['POST'])
@cross_origin()
def label():
    tweet = request.form['tweet']


    l=lstm(tweet)
    lrs=lstm_rs(tweet)
    
    r=0 
 
    if lrs==0:
        r=0
    elif lrs==1:
        r=1
    elif l==0:
        r=2
    elif l==1:
        r=3
    else:
        r=4

    return jsonify({'category': r}), 200




if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)
