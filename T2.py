from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import string
import csv
from sklearn.feature_extraction.text import TfidfVectorizer


train_data = pd.read_csv("../data/train.tsv",delimiter = '\t', quoting=csv.QUOTE_NONE)
test_data = pd.read_csv("../data/test.tsv",delimiter = '\t' , quoting=csv.QUOTE_NONE)

for index,tweet in train_data.iterrows():
    text = tweet['text']
    text = text.lower()
    text = "".join(x for x in text if x not in string.punctuation)
    train_data.loc[index,'text'] = text

vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)
vector = vectorizer.fit_transform(train_data['text'])
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=150)
classifier.fit(vector, train_data['hateful'])

for index,tweet in test_data.iterrows():
    text = tweet['text']
    text = text.lower()
    text = "".join(x for x in text if x not in string.punctuation)
    test_data.loc[index,'text'] = text

testVec = vectorizer.transform(test_data['text'])

yPred = classifier.predict(testVec)

testPred = []
for index,tweet in test_data.iterrows():
    tdict = {}
    tdict['id'] = tweet['id']
    tdict['hateful'] = yPred[index]
    testPred.append(tdict)

fields = ['id','hateful']
t2csv = "../predictions/T2.csv"
with open(t2csv, 'w') as csvfile:  
    writer = csv.DictWriter(csvfile, fieldnames = fields)  
    writer.writeheader()  
    writer.writerows(testPred)

