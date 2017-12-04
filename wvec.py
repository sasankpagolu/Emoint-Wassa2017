import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#print model['sasank']

import numpy as np

#from sklearn.linear_model import SVC
dimension=300

for emotion in ['anger','joy','sadness','fear'] :
    inp='Tweetsdata/'+emotion+'_train_pproc.txt'
    op=emotion+'_train_300'+'.npy'
    out=list()
    fp=open(inp)
    for line in fp:
        vals=line.split('\t')
        count=0
        allsum=np.zeros((dimension,),dtype="float32")
        res=[]
        for word in vals[0].split(' '):
            if word in model.wv.vocab:
                y=model[word]
                allsum=np.add(allsum,y)
                count=count+1

        if count!=0:
            res=np.divide(allsum,count)
        else:
            res=allsum

        out.append(res)


    np.save(op,out)
    fp.close()
    print emotion+'train done'

    inp='Tweetsdata/'+emotion+'_test_pproc.txt'
    op=emotion+'_test_300'+'.npy'
    out=list()
    fp=open(inp)
    for line in fp:
        vals=line.split('\t')
        count=0
        allsum=np.zeros((dimension,),dtype="float32")
        res=[]
        for word in vals[0].split(' '):
           if word in model.wv.vocab:
                y=model[word]
                allsum=np.add(allsum,y)
                count=count+1

        if count!=0:
            res=np.divide(allsum,count)
        else:
            res=allsum

        out.append(res)


    np.save(op,out)
    fp.close()
    print emotion+'test done'
