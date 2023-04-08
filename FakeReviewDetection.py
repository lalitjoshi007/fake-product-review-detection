import numpy as np 
import pandas as pd 
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('stopwords')

dataframe = pd.read_csv('/content/dataset.csv')
dataframe.head()

dataframe.drop('Unnamed: 0',axis=1,inplace=True)
dataframe.head()

dataframe.dropna(inplace=True)                                                                                        # Dropping alll the null rows in the dataset
dataframe['length'] = dataframe['text_'].apply(len)                                                                   # Storing the length of all the text into a separate column called 'length'
dataframe[dataframe['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_ # So here we are just collecting the words which are most common in the fake reviews so that we can identify these words to detect for future text

def convertmyTxt(rv): 
    np = [c for c in rv if c not in string.punctuation]                           # This function is checking if it is present in punctuation or not.
    np = ''.join(np)                                                              # The character which are not in punctuation, we are storing them in a separate string
    return [w for w in np.split() if w.lower() not in stopwords.words('english')] # Here we are returning a list of words from the sentences we just made in above line and checking if it is not a stopword
 
x_train, x_test, y_train, y_test = train_test_split(dataframe['text_'],dataframe['label'],test_size=0.25)

# Here we are defining our RandomForestClassifier model in which we will pass the training and testing data
pip = Pipeline([
    ('bow',CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
]) 
pip.fit(x_train,y_train)

# Here we are predicting the accuracy of the Random Forest Classifier model
randomForestClassifier = pip.predict(x_test) 
randomForestClassifier

print('Accuracy of the RandomForestmodel: ',str(np.round(accuracy_score(y_test,randomForestClassifier)*100,2)) + '%')

# Here we are defining our Support Vector Classifier model in which we will pass the training and testing data
pip = Pipeline([
    ('bow',CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])
pip.fit(x_train,y_train)

# Here we are predicting the accuracy of the Random Forest Classifier model
supportVectorClassifier = pip.predict(x_test)
supportVectorClassifier

print('Accuracy of the SVC model:',str(np.round(accuracy_score(y_test,supportVectorClassifier)*100,2)) + '%')

# Here we are defining our Logistic Regression model in which we will pass the training and testing data
pip = Pipeline([
    ('bow',CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])
pip.fit(x_train,y_train)

# Here we are predicting the accuracy of the Logistic Regression model
logisticRegression = pip.predict(x_test)
logisticRegression

print('Accuracy of the Logistic Regression model:',str(np.round(accuracy_score(y_test,logisticRegression)*100,2)) + '%')