import pandas as pd 
import numpy as np
import string
import csv

train_data = pd.read_csv("../data/train.tsv",delimiter = '\t', quoting=csv.QUOTE_NONE)
test_data = pd.read_csv("../data/test.tsv",delimiter = '\t' , quoting=csv.QUOTE_NONE)

for index,tweet in train_data.iterrows():
	text = tweet['text']
	text = text.lower()
	text = "".join(x for x in text if x not in string.punctuation)
	train_data.loc[index,'text'] = text

#### TfIdf + RF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)
vector = vectorizer.fit_transform(train_data['text'])
rf = RandomForestClassifier()
rf_model = rf.fit(vector, train_data['hateful'])

for index,tweet in test_data.iterrows():
	text = tweet['text']
	text = text.lower()
	text = "".join(x for x in text if x not in string.punctuation)
	test_data.loc[index,'text'] = text

testVec = vectorizer.transform(test_data['text'])
y_pred = rf_model.predict(testVec)
rfPred = []
for index,tweet in test_data.iterrows():
	tdict = {}
	tdict['id'] = tweet['id']
	tdict['hateful'] = y_pred[index]
	rfPred.append(tdict)

fields = ['id','hateful']
rfcsv = "../predictions/RF.csv"
with open(rfcsv, 'w') as csvfile:  
    writer = csv.DictWriter(csvfile, fieldnames = fields)  
    writer.writeheader()  
    writer.writerows(rfPred)

#### word2vec + svm
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
nlp = spacy.load("en_core_web_md")

count = 0
for index,tweet in train_data.iterrows():
    doc = nlp(tweet['text'])
    if(count == 0):
        wvec= np.array([doc.vector])
        label = [tweet['hateful']]
    elif (count == 1):
        wvec = np.concatenate((wvec,[doc.vector]),axis=0)
        label.append(tweet['hateful'])
    else:
        wvec = np.concatenate((wvec,[doc.vector]),axis=0)
        label.append(tweet['hateful'])
    count += 1   

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
clf.fit(wvec,label)
count = 0
for index,tweet in test_data.iterrows():
    doc = nlp(tweet['text'])
    if(count == 0):
        testWVec= np.array([doc.vector])
    elif (count == 1):
        testWVec = np.concatenate((wvec,[doc.vector]),axis=0)
    else:
        testWVec = np.concatenate((wvec,[doc.vector]),axis=0)
    count += 1    

svmY = clf.predict(testWVec)
svmPred = []
for index,tweet in test_data.iterrows():
	tdict = {}
	tdict['id'] = tweet['id']
	tdict['hateful'] = svmY[index]
	svmPred.append(tdict)
svmcsv = "../predictions/SVM.csv"
with open(svmcsv, 'w') as csvfile:  
    writer = csv.DictWriter(csvfile, fieldnames = fields)  
    writer.writeheader()  
    writer.writerows(svmPred)


#### fasttext
from fasttext import train_supervised
fasttrain = open("fasttrain.txt", "w")
for index,tweet in train_data.iterrows():
    fasttrain.write(tweet['text'] + " __label__" + str(tweet['hateful'])+'\n')
model = train_supervised(input="fasttrain.txt")

fastY = []
for index,tweet in test_data.iterrows():
    textin = test_data.loc[index,'text']
    textin = textin.replace('\n','')
    fastY.append(int(model.predict(textin)[0][0][-1]))
fastPred = []
for index,tweet in test_data.iterrows():
	tdict = {}
	tdict['id'] = tweet['id']
	tdict['hateful'] = fastY[index]
	fastPred.append(tdict)
fastcsv = "../predictions/FT.csv"
with open(fastcsv, 'w') as csvfile:  
    writer = csv.DictWriter(csvfile, fieldnames = fields)  
    writer.writeheader()  
    writer.writerows(fastPred)