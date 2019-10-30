import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import re
from nltk.stem import PorterStemmer
import scipy.io as sio

def inputfile(string = None):
    str = string
    if not string:
        str = input("enter the file name: ")
    readfile = "Spam Classifier/data/" + str
    return readfile


def Readtxtfile(string):
    contents = ""
    with open(string, 'r') as fileinput:
        for line in fileinput:
            contents += line.lower()
    return contents

def getvocablist():
    f = open("Spam Classifier/data/vocab.txt", 'r')
    vlist =  f.readlines()
    vlist = [re.sub('[0-9]+','',i) for i in vlist]
    vlist = [i.strip() for i in vlist]
    return vlist


def split(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def processEmail(email_contents):
    p = re.compile('<[^<>]+')
    email_contents = p.sub(" ", email_contents)
    p = re.compile('[0-9]+')
    email_contents = p.sub('number', email_contents)
    p = re.compile('(http|https)://[^\s]*')
    email_contents = p.sub('httpaddr', email_contents)
    p = re.compile('[^\s]+@[^\s]+')
    email_contents = p.sub('emailaddr', email_contents)
    p = re.compile('[$]+')
    email_contents = p.sub('dollars', email_contents)

    delimters = "@","$","/","#",".","-",":","&","*","+","=","[","]","?","!","(",")","{","}",",",">","_","<",";","%","\n"," "
    email_contents = split(delimters,email_contents)
    stemmer = PorterStemmer()
    email_contents = [stemmer.stem(elem) for elem in email_contents]
    word_indices = []
    for elem in email_contents:
        if len(elem) < 1:
            continue
        for i in range(len(vocablist)):
            if elem == vocablist[i]:
                word_indices.append(i)
                break

    return word_indices


def get_features(size,word_indices):
    n = size
    result = np.zeros(n)
    for i in range(len(word_indices)):
        result[word_indices[i]] = 1

    return result


raw_data_train = sio.loadmat(inputfile("spamTrain.mat"))
X_train = raw_data_train['X']
y_train = raw_data_train['y']


clf = SVC(kernel="linear", C=1)
clf.fit(X_train,y_train.ravel())
predict = clf.predict(X_train)
print(np.mean(y_train.ravel() == predict.ravel()))

raw_data_test = sio.loadmat(inputfile("spamTest.mat"))
X_test = raw_data_test['Xtest']
y_test = raw_data_test['ytest']
predict_test = clf.predict(X_test)
print(np.mean(y_test.ravel() == predict_test.ravel()))

vocablist = getvocablist()
vocalist_size = len(vocablist)

sorted = np.argsort(clf.coef_)
sorted = sorted[::-1].ravel()
for i in range(10):
    print(vocablist[int(sorted[i])])



email_contents = Readtxtfile(inputfile("spamSample2.txt"))
word_indices = processEmail(email_contents)
features = get_features(vocalist_size, word_indices)
features = features.reshape(1, -1)
is_spam = clf.predict(features)
if is_spam:
    print("spam")
else:
    print("no")


