import numpy as np



fp=open('/home/sasank/nlp/glove.twitter.27B/glove.twitter.27B.100d.txt')
dimension=100
glove=dict()
i=0
for line in fp:
	values=line.strip('\n').split(' ')
	glove[values[0]]=values[1:len(values)]
	i=i+1
	if i%100000==0:
		print i
fp.close()


for emotion in ['anger','joy','sadness','fear'] :
    inp='Tweetsdata/'+emotion+'_train_pproc.txt'
    op=emotion+'_train_100'+'.npy'
    out=list()
    fp=open(inp)
    for line in fp:
        vals=line.split('\t')
        count=0
        allsum=np.zeros((dimension,),dtype="float32")
        for word in vals[0].split(' '):
            x=glove.get(word)
            if x is not None:
                y=[float(v) for v in x]
                allsum=np.add(allsum,y)
                count=count+1

        if count!=0:
            allsum=np.divide(allsum,np.max(allsum))

        out.append(allsum)

    np.save(op,out)
    fp.close()
    print emotion+'train done'

    inp='Tweetsdata/'+emotion+'_test_pproc.txt'
    op=emotion+'_test_100'+'.npy'
    out=list()
    fp=open(inp)
    for line in fp:
        vals=line.split('\t')
        count=0
        allsum=np.zeros((dimension,),dtype="float32")
        for word in vals[0].split(' '):
            x=glove.get(word)
            if x is not None:
                y=[float(v) for v in x]
                allsum=np.add(allsum,y)
                count=count+1

        if count!=0:
            allsum=np.divide(allsum,np.max(allsum))

    
        out.append(allsum)





    np.save(op,out)
    fp.close()
    print emotion+'test done'

'''
    score_train=list()
    fp=open('Tweetsdata/anger_train_pproc.txt')
    for line in fp:
        values= line.split('\t')
        score_train.append(float(values[1]))

    fp.close()
    score_gold=list()
    fp=open('Tweetsdata/anger_test_pproc.txt')
    for line in fp:
        values= line.split('\t')
        score_gold.append(float(values[1]))

    fp.close()

    ml_model = LogisticRegression
    print X_train
    ml_model.fit(X_train, score_train)
    y_pred = ml_model.predict(X_test)

#score = evaluate_lists(y_pred, score_gold)
    print y_pred,score_gold
    pears_corr = scipy.stats.pearsonr(y_pred, score_gold)[0]
    spear_corr = scipy.stats.spearmanr(y_pred, score_gold)[0]

    print pears_corr,spear_corr
'''



    