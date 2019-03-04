from tokenizer import tokenize
import numpy as np
import argparse
#import gensim
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, LSTM
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import model_from_json

embedding_size=200

def sent_index(tweets):
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

def get_embedding():
    f=open("embedd.200d_mix.pickle","rb")
    embedd=pickle.load(f)
    w=np.zeros([voc_count,embedding_size])
    for k,v in vocab_index.items():
        w[v]=embedd[k]

    return(w)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(voc_count, embedding_dim, input_length=sequence_length, trainable=True))
    model.add(Dropout(0.35))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(90))
    model.add(Dropout(0.35))
    model.add(Dense(7))
    model.add(Dense(6))
    model.add(Dense(5))
    model.add(Dense(4))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    '''
    return(model)

def fast_text_model(sequence_length, embedding_dim):
    model = Sequential()
    model.add(Embedding(voc_count, embedding_dim, input_length=sequence_length, trainable=True))
    #model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, input_length=sequence_length, trainable=False))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return(model)

def cnn_model(sequence_length, embedding_dim):
    filters=(3,4,5)
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs=[]
    for f in filters:
        conv = Convolution1D(nb_filter=100,
                             filter_length=f,
                             border_mode='valid',
                             activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)

        convs.append(pool)
    if len(filters)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    model = Sequential()
    model.add(Embedding(voc_count, embedding_dim, input_length=sequence_length,trainable=True))
    model.add(Dropout(0.25))
    model.add(graph)
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(7))
    model.add(Dense(6))
    model.add(Dense(5))
    model.add(Dense(4))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    return(model)

f=open("vocab_index_mix.pickle","rb")
vocab_index=pickle.load(f)
voc_count=len(vocab_index)

tweets=np.load("tweets_mix.npy")
x=sent_index(tweets)
print(x)
y=np.load("labels_mix.npy")
x, y = shuffle(x, y, random_state=9)
#x, y = shuffle(x, y, random_state=9)
Y=y
y = np_utils.to_categorical(y, 3)

'''
T=np.load("tweets_np_.npy")[:100]
#T=T[700000:]
x_T=sent_index(T)
'''
#xt=np.concatenate((x_T,x))

max_len = max(map(lambda i:len(i), x))
x = pad_sequences(x, maxlen=max_len)
#x_T = pad_sequences(x_T, maxlen=max_len)

w=get_embedding()
rand_w=np.random.rand(voc_count,embedding_size)

model=cnn_model(max_len,embedding_size)
print(model.summary())

model.layers[0].set_weights([rand_w])

#n=18347
#nit=16650
#nnp=18924
n=int(0.8*len(tweets))
#n=16650
#n=80
model.fit(x[:n], y[:n], epochs=9, batch_size=128)
#model.fit(x, y, epochs=6, batch_size=128)
#scores = model.evaluate(x[n:], y[n:], verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred=model.predict_on_batch(x[n:])
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(Y[n:],y_pred))

'''
y_pred=model.predict_on_batch(x_T)
y_pred = np.argmax(y_pred, axis=1)

Y_pred=np.load("labels_np_.npy")[:100]

print(classification_report(Y_pred,y_pred))
'''
#np.save("result_itnp.npy",y_pred)


#embedd_w=model.layers[0].get_weights()
#np.save("emd_w_LSTM_glove.npy",embedd_w)
#np.save("emd_w_LSTM_rand.npy",embedd_w)

print("Saved embedding weights")

# serialize model to JSON
model_json = model.to_json()
with open("model_.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_.h5")
print("Saved model to disk")


# load json and create model
json_file = open('model_.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x[n:], y[n:], verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print(max_len)








