# -*- coding: utf-8 -*-
"""
Importing data from UN COMTRADE to identify countries which are net exporters of 
coffee (non-roasted, non-decaffeinated)
"""
from urllib2 import Request, urlopen

request=Request('http://comtrade.un.org/api/get?ps=2014&fmt=json&px=HS&p=0&r=all&cc=090111')
response = urlopen(request)
data = json.loads(response.read())
data = json_normalize(data['dataset'])

"""
Importing U.S. import data of coffee, downloaded as CSV from USITC http://dataweb.usitc.gov/,
and national coffee production data from NationMaster http://www.nationmaster.com/country-info/stats/Agriculture/Crops/Beans/Coffee/Coffee-production
"""
import pandas as pd
imports = pd.read_csv('USITC.csv')


national_output = pd.read_csv('national coffee output.csv')


"""
Importing percent coffee certified by country, from PDF report: https://www.iisd.org/pdf/2014/ssi_2014_chapter_8.pdf
"""
cert = pd.read_csv('cert.csv')
