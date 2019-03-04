import numpy as np
import pickle

# code to generate vocab_index 

#f=open("embedd.200d_np.pickle","rb")
f=open("embedd.200d_mix.pickle","rb")
emd=pickle.load(f)
#e=np.load("vocab_np.npy")
#e=np.load("vocab_mix.npy")
#it=np.load("vocab_it.npy")
#e=np.concatenate((ne,it))
n=0
d={}
for k,v in emd.items():
    d[k]=n
    n+=1

#f=open("vocab_index_np_.pickle","wb")
f=open("vocab_index_mix.pickle","wb")
pickle.dump(d,f)
print("Done...saved")


