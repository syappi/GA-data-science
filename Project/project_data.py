# -*- coding: utf-8 -*-
"""
Importing data from UN COMTRADE to identify countries which are net exporters of 
coffee (non-roasted, non-decaffeinated)
"""
from urllib2 import Request, urlopen
import json
from pandas.io.json import json_normalize
import pandas as pd

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
data = data[data['Custom Value 2014'] != '0']
data = data.reset_index()

"Preparing X"

national_output = pd.read_csv('national coffee output.csv')


"""
Importing percent coffee certified by country, from PDF report: https://www.iisd.org/pdf/2014/ssi_2014_chapter_8.pdf
"""
cert = pd.read_csv('cert.csv')
