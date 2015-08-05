"""
Importing data from UN COMTRADE to identify countries which are net exporters of 
coffee (non-roasted, non-decaffeinated)
"""
from urllib2 import Request, urlopen
import json
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np

request=Request('http://comtrade.un.org/api/get?ps=2014&fmt=json&px=HS&p=0&r=all&cc=090111')
response = urlopen(request)
data = json.loads(response.read())
data = pd.DataFrame(json_normalize(data['dataset']))

countries = data.rtTitle.unique()
qualifier = pd.DataFrame({'countries': countries, 'import': 0, 'export': 0, 're_export': 0})

i=0

for x in countries:
    if data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Import')].TradeQuantity.empty == False:
        qualifier.iloc[i,2] = float(data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Import')].TradeQuantity)
    else: qualifier.iloc[i,2] = 0    
    
    if data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Export')].TradeQuantity.empty == False:
        qualifier.iloc[i,1] = float(data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Export')].TradeQuantity)
    else: qualifier.iloc[i,1] = 0
        
    if data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Re-Export')].TradeQuantity.empty == False:
        qualifier.iloc[i,3] = float(data[(data.rtTitle == countries[i]) & (data.rgDesc == 'Re-Export')].TradeQuantity)
    else: qualifier.iloc[i,3] = 0
    i=i+1


qualifier['net_export'] = qualifier['export'] + qualifier['re_export'] - qualifier['import']

"""
Countries excluded from analysis, not net exporters of coffee
"""

exclude = qualifier[qualifier.net_export < 0]
exclude.reset_index()
df = exclude.set_index(['countries'])

"""
Importing U.S. import data of coffee, downloaded as CSV from USITC http://dataweb.usitc.gov/,
and national coffee production data from NationMaster http://www.nationmaster.com/country-info/stats/Agriculture/Crops/Beans/Coffee/Coffee-production
"""
import pandas as pd
imports = pd.read_csv('USITC.csv')
imports = imports[imports['HTS Number'] == 9011100]

"Dropping countries which are not net exporters"
imports['delete'] = 0
i=0
for x in imports['Country']:
    if (imports.iloc[i,2] in df.index.values) == True:
        imports.iloc[i,11] = 1
    else: imports.iloc[i,11] = 0
    i = i+1


data = imports[imports.delete != 1]
data = data[(data['Custom Value 2014'] != 0) & (data['Custom Value 2011'] != 0)]
data = data.reset_index()

"Rename country names"
data.replace({'Congo (DROC)' : 'Democratic Republic of the Congo', 'Congo (ROC)': 'Republic of the Congo', 'Dominican Rep': 'Dominican Republic'}, inplace=True)

"Remove %"
data['Quantity - Percent Change 2011 - 2014'] = data['Quantity - Percent Change 2011 - 2014'].str.replace('%','')
data['Quantity - Percent Change 2011 - 2014'] = data['Quantity - Percent Change 2011 - 2014'].convert_objects(convert_numeric=True)
data['Avg Price 2014'] = float(0)
data['Avg Price 2011'] = float(0)
data['Avg Price - Percent Change 2011 - 2014'] = float(0)
data['Percent Market Share'] = float(0)

for index, x, in enumerate(data['Country']):
    data['Avg Price 2014'][index] = float(data['Custom Value 2014'][index])/data['Quantity 2014'][index]
    data['Avg Price 2011'][index] = float(data['Custom Value 2011'][index])/data['Quantity 2011, kg'][index]
    data['Avg Price - Percent Change 2011 - 2014'][index] = float(data['Avg Price 2014'][index] - data['Avg Price 2011'][index])/data['Avg Price 2011'][index]*100
    data['Percent Market Share'][index] = float(data['Custom Value 2014'][index])/np.sum(data['Custom Value 2014'])*100


"Import total coffee production of individual countries"

national_output = pd.read_csv('national coffee output.csv')
national_output = national_output.iloc[:,0:2]
national_output.dropna(inplace=True)
national_output = national_output.set_index(['Country'])

"Join U.S. import data of coffee by exporting country with total coffee production"

full_data = data.join(national_output, on=['Country'])
full_data.fillna(value=0, inplace=True)

"""
Importing percent coffee certified by country, from PDF report: https://www.iisd.org/pdf/2014/ssi_2014_chapter_8.pdf
"""
cert = pd.read_csv('cert.csv')
cert['Adjusted Aggregate*'] = cert['Adjusted Aggregate*'].str.replace('[^\w\s]','')
cert.rename(columns={'Adjusted Aggregate*':'Percent Certified'}, inplace=True)
cert.fillna(value=0, inplace=True)
cert = cert.set_index(['Country'])
cert = cert.iloc[:,5]
cert = cert.convert_objects(convert_numeric=True)

full_data = full_data.join(cert, on=['Country'])
full_data.fillna(value=0, inplace=True)
full_data = full_data.set_index(['Country'])

"Correlation Matrix"

import seaborn as sns
import matplotlib.pyplot as plt

plot_cols = ['Avg Price 2014', 'Avg Price - Percent Change 2011 - 2014', 'Quantity - Percent Change 2011 - 2014', 'Output_million_kg', 'Percent Certified', 'Percent Market Share']
data = full_data[plot_cols]
data.corr()
sns.heatmap(data.corr())
sns.pairplot(data, x_vars=['Avg Price 2014', 'Avg Price - Percent Change 2011 - 2014', 'Quantity - Percent Change 2011 - 2014', 'Output_million_kg', 'Percent Certified'], y_vars='Percent Market Share', size=6, aspect=0.7, kind='reg')
feature_cols = ['Avg Price 2014', 'Avg Price - Percent Change 2011 - 2014', 'Quantity - Percent Change 2011 - 2014', 'Output_million_kg', 'Percent Certified']
X = full_data[feature_cols]
y = full_data['Percent Market Share']

"Plotting 'Quantity - Percent Change 2011 - 2014' (< +100%) vs Percent Market Share"

quant100 = full_data[full_data['Quantity - Percent Change 2011 - 2014'] < 100]['Quantity - Percent Change 2011 - 2014']
y_quant100 = full_data[full_data['Quantity - Percent Change 2011 - 2014'] < 100]['Percent Market Share']
m, b = np.polyfit(quant100, y_quant100, 1)
plt.plot(quant100, y_quant100, '.')
plt.plot(quant100, m*quant100 + b, '-')
plt.xlabel('Quantity - Percent Change 2011 - 2014, less than +100%')
plt.ylabel('Percent Market Share')
plt.axis([-100, 100, -5, 30])
plt.annotate('Positive Correlation', xy=(50, 6), xytext=(65, 8),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

"Linear Regression"
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split, cross_val_score

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred)), np.mean(np.sqrt(-scores))

train_test_rmse(X, y)

"Regression without Price information"

feature_cols = ['Quantity - Percent Change 2011 - 2014', 'Output_million_kg', 'Percent Certified']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Output_million_kg', 'Percent Certified']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

"Regression with Price information"

feature_cols = ['Avg Price 2014', 'Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Avg Price 2014', 'Percent Certified', 'Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Avg Price 2014', 'Quantity - Percent Change 2011 - 2014', 'Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Avg Price 2014', 'Avg Price - Percent Change 2011 - 2014', 'Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

feature_cols = ['Avg Price 2014', 'Quantity - Percent Change 2011 - 2014', 'Percent Certified', 'Output_million_kg']
X = full_data[feature_cols]
train_test_rmse(X, y)

"Decision Tree for regression #7"

from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
feature_cols = ['Avg Price 2014', 'Avg Price - Percent Change 2011 - 2014', 'Quantity - Percent Change 2011 - 2014', 'Output_million_kg', 'Percent Certified']
X = full_data[feature_cols]
treereg = DecisionTreeRegressor(random_state=1)
max_depth_range = range(1, 11)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(treereg, param_grid, cv=10, scoring='mean_squared_error')
grid.fit(X, y)

grid.best_params_
grid.best_estimator_

from sklearn.tree import export_graphviz
import pydot
treeregbest = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=1)
treeregbest.fit(X, y)

export_graphviz(treeregbest, out_file='tree_coffee.dot', feature_names=feature_cols)
graph = pydot.graph_from_dot_file('tree_coffee.dot')
graph.write_png('tree_coffee.png')
