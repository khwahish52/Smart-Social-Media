'''
import numpy as np
import pickle
import gensim

# code to generate embedding file for fast embedding

model=gensim.models.KeyedVectors.load_word2vec_format("/home/arindam/Desktop/tweet_classification/glove/word2vec.glove.twitter.27B.200d.txt")
embedd={}
vocab=np.load("vocab_macy.npy")

n=0
for v in vocab:
    try:
        embedd[v]=np.random.rand(1,200)[0]
        n+=1
        print("Words : "+str(n)+"/"+str(len(vocab)))
    except:
        print(v+" not in embedding")  

print("Done getting embedding")
emdf=open("embedd.200d_macy.pickle","wb")
pickle.dump(embedd,emdf)
print("Done Saving")
'''
import numpy as np
import pickle
import gensim

# code to generate embedding file for fast embedding

model=gensim.models.KeyedVectors.load_word2vec_format("/home/arindam/Desktop/tweet_classification/glove/word2vec.glove.twitter.27B.200d.txt")
embedd={}
vocab=np.load("vocab_mix.npy")

n=0
for v in vocab:
    try:
        embedd[v]=model.wv[v]
        n+=1
        print("Words : "+str(n)+"/"+str(len(vocab)))
    except:
        print(v+" not in embedding")  

print("Done getting embedding")
emdf=open("embedd.200d_mix.pickle","wb")
pickle.dump(embedd,emdf)
print("Done Saving")

