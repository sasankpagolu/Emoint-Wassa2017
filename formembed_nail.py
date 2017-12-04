import numpy as np

#from sklearn.linear_model import SVC
emotionmap=dict()
emotionmap['anger']=0
emotionmap['fear']=1
emotionmap['sadness']=2
emotionmap['joy']=3
lex=[dict() for x in range(4)]
fp=open('Tweetsdata/nail.txt')
for line in fp:
    values=line.strip('\n').split('\t')
    lex[emotionmap[values[2]]][values[0]]=values[1]
fp.close()

fp=open('/home/sasank/nlp/Tweetsdata/hashtag.txt')
lex2=[dict() for x in range(4)]
for line in fp:
    values=line.strip('\n').split('\t')
    x=emotionmap.get(values[0])
    if x is not None:
        lex2[emotionmap[values[0]]][values[1].strip('#')]=values[2]
fp.close()

fp=open('/home/sasank/nlp/Tweetsdata/nelex.txt')
lex3=dict()
for line in fp:
    values=line.strip('\n').split('\t')
    x=emotionmap.get(values[0])
    if x is not None:
        lex3[values[0].strip('#')]=values[1]
fp.close()

np.save('dict1_anger.npy',lex[emotionmap['anger']])
np.save('dict1_fear.npy',lex[emotionmap['fear']])
np.save('dict1_sadness.npy',lex[emotionmap['sadness']])
np.save('dict1_joy.npy',lex[emotionmap['joy']])
np.save('dict2_anger.npy',lex2[emotionmap['anger']])
np.save('dict2_fear.npy',lex2[emotionmap['fear']])
np.save('dict2_sadness.npy',lex2[emotionmap['sadness']])
np.save('dict2_joy.npy',lex2[emotionmap['joy']])
np.save('dict3.npy',lex3)

fp=open('glove.twitter.27B/glove.twitter.27B.100d.txt')
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
    op=emotion+'_train_109'+'.npy'
    out=list()
    fp=open(inp)
    hw=0
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
            else:
                hw=hw+1

        if count!=0:
            allsum=np.divide(allsum,count)

        count=0
        sumem=0
        emanger=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['anger']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emanger=sumem/count
        else:
            emanger=sumem

        count=0
        sumem=0
        emfear=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emfear=sumem/count
        else:
            emfear=sumem

        count=0
        sumem=0
        emsadness=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emsadness=sumem/count
        else:
            emsadness=sumem

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem

        n1=np.append(emanger,emfear)
        n2=np.append(n1,emsadness)
        n3=np.append(n2,emjoy)

        count=0
        sumem=0
        emanger=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['anger']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emanger=sumem/count
        else:
            emanger=sumem

        count=0
        sumem=0
        emfear=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emfear=sumem/count
        else:
            emfear=sumem

        count=0
        sumem=0
        emsadness=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emsadness=sumem/count
        else:
            emsadness=sumem

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem

        mn1=np.append(emanger,emfear)
        mn2=np.append(n1,emsadness)
        mn3=np.append(n2,emjoy)
        fn=np.append(n3,mn3)

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex3.get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem
        ffn=np.append(fn,emjoy)
        out.append(np.append(allsum,ffn))


    np.save(op,out)
    fp.close()
    print emotion+'train done',hw

    inp='Tweetsdata/'+emotion+'_test_pproc.txt'
    op=emotion+'_test_109'+'.npy'
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
            allsum=np.divide(allsum,count)

        count=0
        sumem=0
        emanger=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['anger']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emanger=sumem/count
        else:
            emanger=sumem

        count=0
        sumem=0
        emfear=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emfear=sumem/count
        else:
            emfear=sumem

        count=0
        sumem=0
        emsadness=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emsadness=sumem/count
        else:
            emsadness=sumem

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem

        n1=np.append(emanger,emfear)
        n2=np.append(n1,emsadness)
        n3=np.append(n2,emjoy)

        count=0
        sumem=0
        emanger=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['anger']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emanger=sumem/count
        else:
            emanger=sumem

        count=0
        sumem=0
        emfear=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emfear=sumem/count
        else:
            emfear=sumem

        count=0
        sumem=0
        emsadness=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emsadness=sumem/count
        else:
            emsadness=sumem

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex2[emotionmap['fear']].get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem

        mn1=np.append(emanger,emfear)
        mn2=np.append(n1,emsadness)
        mn3=np.append(n2,emjoy)
        fn=np.append(n3,mn3)

        count=0
        sumem=0
        emjoy=0
        for word in vals[0].split(' '):
            x1=lex3.get(word)
            if x1 is not None:
                #if x1[1]==emotion:
                sumem=sumem+float(x1)
                count=count+1

        if count!=0:
            emjoy=sumem/count
        else:
            emjoy=sumem
        ffn=np.append(fn,emjoy)
        out.append(np.append(allsum,ffn))

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



    