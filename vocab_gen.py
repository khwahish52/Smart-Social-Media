import numpy as np
import pickle
from tokenizer import tokenize

# code to generate vocabulary

vocab=[]
tweets=np.load("tweets_mix.npy")
n=0
for t in tweets:
    words=tokenize(t)
    for w in words:
        if w not in vocab:
            vocab.append(w)
    n+=1
    print("Tweets : "+str(n)+"/16907")


np.save("vocab_mix.npy",vocab)

