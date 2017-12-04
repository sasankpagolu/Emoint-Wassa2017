#New Approach

import pandas as pd
import numpy as np
#import lightgbm as lgb
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import nltk

from xgboost import XGBClassifier

import os

from nltk.tokenize import word_tokenize

for emotion in['anger','sadness','joy','fear']:
# Read training file
    df_train = pd.read_csv('Tweetsdata/'+emotion+'_train_pproc.txt', sep='\t', names=["Text","probs"])
    df_train.head()


    df_test = pd.read_csv('Tweetsdata/'+emotion+'_test_pproc.txt', sep='\t', names=["Text","probs"])
    df_test.head()

    print("reading done")


    count_vectorizer_text = TfidfVectorizer( strip_accents ='ascii',
    analyzer="word", tokenizer=nltk.word_tokenize,stop_words='english',
    max_features=None,ngram_range=(1,3))



    tfidf_train_text = count_vectorizer_text.fit_transform(df_train['Text'].values.astype('U'))

    tfidf_test_text = count_vectorizer_text.fit_transform(df_test['Text'].values.astype('U'))
    print len(count_vectorizer_text.vocabulary_)
    svd_text = TruncatedSVD(n_components= 500)
    tfidf_train_text = svd_text.fit_transform(tfidf_train_text)
    tfidf_test_text = svd_text.fit_transform(tfidf_test_text)


    print('Step::2 Done')

    features_train = tfidf_train_text
    features_test = tfidf_test_text
    
    np.save('tfidffeatures_train_'+emotion+'.npy',features_train)
    np.save('tfidffeatures_test_'+emotion+'.npy',features_test)

'''

#-----------#
xgb = XGBClassifier(max_depth=4,
                          objective='multi:softprob',
                          learning_rate=0.03333)
xgb.fit(X, y)
probas = xgb.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.to_csv('submission_xgb.csv', index=False)

#----------------#
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X, y)
probas = rf.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()
submission_df.to_csv('submission_rf.csv', index=False)

#---------#

print(X.shape)
gbm = lgb.LGBMClassifier(objective='multiclass',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X, y)
probas = gbm.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()
submission_df.to_csv('submission_gbm.csv', index=False)

# probas_train = gbm.predict_proba(features_train)
# submission_df = pd.DataFrame(probas_train, columns=['class'+str(c+1) for c in range(9)])
# submission_df['ID'] = df_test['ID']
# submission_df.head()
# submission_df.to_csv('submission_gbm_train.csv', index=False)
# pred_indices = np.argmax(probas_train, axis=1)
# classes = np.unique(y)
# preds = classes[pred_indices]

# print('Log loss: {}'.format(log_loss(y, probas_train)))
# print('Accuracy: {}'.format(accuracy_score(y, preds)))
'''