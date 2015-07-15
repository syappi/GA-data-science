# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:41:54 2015

@author: Scott
"""

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/titanic.csv'
titanic = pd.read_csv(url, index_col='PassengerId')

"""
Define Pclass and Parch as the features, and Survived as the response.
"""

feature_cols = ['Pclass', 'Parch']
X = titanic[feature_cols]
y = titanic.Survived

"""
Split the data into training and testing sets.
Fit a logistic regression model and examine the coefficients to confirm that they make intuitive sense.
"""
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)


"""
Make predictions on the testing set and calculate the accuracy.
"""
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


"""
Bonus: Compare your testing accuracy to the "null accuracy", a term we've seen once before.
Bonus: Add Age as a feature, and calculate the testing accuracy. There will be a small issue you'll have to deal with.
"""