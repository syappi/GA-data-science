# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:40:15 2015

@author: Setra
"""

"""
Read yelp.csv
"""
import pandas as pd
yelp = pd.read_csv('yelp.csv')
yelp.head()

"""
Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=6, aspect=0.7)

"""
Define cool/useful/funny as the features, and stars as the response.
"""
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

"""
Fit a linear regression model and interpret the coefficients. 
Do the coefficients make intuitive sense to you? 
Explore the Yelp website to see if you detect similar trends.
ANS: 
[('cool', 0.27435946858852989),
 ('useful', -0.14745239099401236),
 ('funny', -0.13567449053706199)]
Yes, cool ranges from 0-77, useful 0-76, funny 0-57. To get a response of 0-5 the coefficients make sense.
"""
linreg = LinearRegression()
linreg.fit(X, y)

print linreg.intercept_
zip(feature_cols, linreg.coef_)

import statsmodels.formula.api as smf
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
print lm.pvalues

"""
Evaluate the model by splitting it into training and testing sets and computing the RMSE. 
Does the RMSE make intuitive sense to you?
ANS: Yes, but the error is quite large -- 1.184 -- for a response within a range 0-5.
"""
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
X = yelp[feature_cols]
train_test_rmse(X, y)
    
"""
Try removing some of the features and see if the RMSE improves.
"""
feature_cols = ['cool', 'useful']
X = yelp[feature_cols]
train_test_rmse(X, y)

feature_cols = ['cool', 'funny']
X = yelp[feature_cols]
train_test_rmse(X, y)

feature_cols = ['cool']
X = yelp[feature_cols]
train_test_rmse(X, y)

feature_cols = ['useful', 'funny']
X = yelp[feature_cols]
train_test_rmse(X, y)


"""
With only cool & useful included in the model, the error increased to 1.196.
cool & funny to 1.194
cool only to 1.21, as well as useful & funny
"""



"""
Bonus: Think of some new features you could create from the existing data that might be predictive of the response. 
(This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, 
and see if the RMSE improves.
"""
yelp.describe()
yelp.drop('stars', axis=1).plot(kind='box')

"""
Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", 
which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.
"""

"""
Bonus: Instead of treating this as a regression problem, treat it as a classification problem 
and see what testing accuracy you can achieve with KNN.
"""

"""
Bonus: Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.
"""
