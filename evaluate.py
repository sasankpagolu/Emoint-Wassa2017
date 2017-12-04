from xgboost import XGBRegressor
import numpy as np
import scipy
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import SVC

def run_test(x_train, score_train, x_test, y_gold):
    ml_model = XGBRegressor(max_depth=3, n_estimators=30000, seed=0)

    ml_model.fit(x_train, score_train)

    y_pred = ml_model.predict(x_test)

    score = evaluate_lists(y_pred, y_gold)
    #print("### " + emotion + ", feature-string: " + feature_string)
    print("| pears-corr | spear-corr | pears-corr-range-05-1 | spear-corr-range-05-1 |")
    print("| --- | --- | --- | --- |")
    print("| " + str(score[0]) + " | " + str(score[1]) + " | " + \
      str(score[2]) + " | " + str(score[3]) + " |")


def evaluate_lists(pred, gold):
    if len(pred) == len(gold):
        gold_scores = gold
        pred_scores = pred

        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1 = []
        pred_scores_range_05_1 = []

        for i in range(len(gold_scores)):
            if(gold_scores[i] >= 0.5):
                gold_scores_range_05_1.append(gold_scores[i])
                pred_scores_range_05_1.append(pred_scores[i])

        '''# return zero correlation if predictions are constant
        if np.std(pred_scores) == 0 or np.std(gold_scores) == 0:
            return (0, 0, 0, 0)
'''
        

        pears_corr_range_05_1 = scipy.stats.pearsonr(pred_scores_range_05_1, gold_scores_range_05_1)[0]
        spear_corr_range_05_1 = scipy.stats.spearmanr(pred_scores_range_05_1, gold_scores_range_05_1)[0]

        return np.array([pears_corr, spear_corr, pears_corr_range_05_1, spear_corr_range_05_1])
    else:
        raise ValueError('Predictions and gold data have different number of lines.')


for emotion in ['anger','fear','sadness','joy']:
    inp1='Tweetsdata/'+emotion+'_train_pproc.txt'
    inp2=emotion+'_train_108.npy'
    inp3=emotion+'_test_108.npy'
    inp4='Tweetsdata/'+emotion+'_test_pproc.txt'

    x_train=np.load(inp2)
    x_test=np.load(inp3)
    score_train=list()

    fp=open(inp1)

    for line in fp:
        values= line.split('\t')
        score_train.append(float(values[1]))
    

    score_gold=list()

    fp=open(inp4)

    for line in fp:
        values= line.split('\t')
        score_gold.append(float(values[1]))

    ml_model=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    ml_model.fit(x_train,score_train)
    y_pred = ml_model.predict(x_test)
    pears_corr = scipy.stats.pearsonr(y_pred, score_gold)[0]
    spear_corr = scipy.stats.spearmanr(y_pred, score_gold)[0]
    print emotion
    print pears_corr,spear_corr
'''
    #GradientBoostingRegressor
    ml_model=GradientBoostingRegressor(n_estimators=1000,n_jobs=-1)
    ml_model.fit(X,score_train)
    y_pred = ml_model.predict(Y)
    pears_corr = scipy.stats.pearsonr(y_pred, score_gold)[0]
    spear_corr = scipy.stats.spearmanr(y_pred, score_gold)[0]
    print emotion
    print pears_corr,spear_corr
'''