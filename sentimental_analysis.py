#Importing Libraries

import numpy as np #to perform array
import pandas as pd #useful for loading the dataset
import re #Regular expressions
import nltk  #Natural Language toolkit
import matplotlib.pyplot as plt #useful for plottinng the graphical data

from nltk.corpus import stopwords #used for removing words like a,an,the
from sklearn.feature_extraction.text import TfidfVectorizer #used for feature extraction
from sklearn.ensemble import RandomForestClassifier #model used for prediction
from sklearn.metrics import accuracy_score #used to check accuracy for all Train & Test Data.
from sklearn.model_selection import train_test_split #Splitting the data

#Load Dataset

dataset = pd.read_csv('dataset.csv')
dataset

#Summarize Dataset

print(dataset.shape)
print(dataset.head(5))

#Segregating Dataset into Input & Output

X = dataset.iloc[:, 10].values #X-->features / Input
X

Y = dataset.iloc[:, 1].values #Y-->labels/output
Y

#Removing the Special Character

features = []

for i in range(0, len(X)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(X[i]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    features.append(processed_feature)
    
    
#Feature Extraction from text

nltk.download('stopwords')
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
features = vectorizer.fit_transform(features).toarray()
print(features)

#Splitting Dataset into Train & Test

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size=0.2, random_state=0)

#Loading Random Forest Algorithm

clf = RandomForestClassifier(n_estimators=200, random_state=0)
clf.fit(X_train, Y_train)

# Accuracy for all Train & Test Data.

pred_tndata = clf.predict(X_train) #prediction on training data
tn_accuracy = accuracy_score(Y_train,pred_tndata)#checking accuracy of training data
print("Accuracy on training data :",tn_accuracy*100,"%")

pred_tsdata =clf.predict(X_test) #prediction on testing data
ts_accuracy = accuracy_score(Y_test,pred_tsdata)
print("Accuracy on testing data :",ts_accuracy*100,"%")


#Plotting predicted data and true data as confusion matrix

from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(Y_test, pred_tsdata, labels=['negative', 'neutral', 'positive'])
plot_confusion_matrix(cm, classes=['negative', 'neutral', 'positive'])
