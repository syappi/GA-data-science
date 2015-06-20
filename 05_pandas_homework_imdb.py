'''
Pandas Homework with IMDB data
'''

'''
BASIC LEVEL
'''
import pandas as pd
import matplotlib.pyplot as plt
# read in 'imdb_1000.csv' and store it in a DataFrame named movies
movies = pd.read_csv('imdb_1000.csv')

# check the number of rows and columns
movies.shape

# check the data type of each column
movies.dtypes

# calculate the average movie duration
movies.duration.mean()

# sort the DataFrame by duration to find the shortest and longest movies
movies.sort('duration')
print 'The longest movie is',movies.sort('duration').tail(1).title.values[0]
print 'The shortest movie is',movies.sort('duration').head(1).title.values[0]

# create a histogram of duration, choosing an "appropriate" number of bins
movies.duration.plot(kind='hist', bins=[50, 100, 150, 200, 250])

# use a box plot to display that same data
movies.duration.plot(kind='box')

'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings
movies.content_rating.notnull().sum()

# use a visualization to display that same data, including a title and x and y labels
movies.content_rating.value_counts().plot(kind='bar', title='Bar Plot of Movies by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Frequency')

# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP
movies.content_rating.replace(['NOT RATED', 'APPROVED', 'PASSED', 'GP'], 'UNRATED', inplace = True)

# convert the following content ratings to "NC-17": X, TV-MA
movies.content_rating.replace({'X': 'NC-17', 'TV-MA': 'NC-17'}, inplace = True)

# count the number of missing values in each column
movies.isnull().sum()

# if there are missing values: examine them, then fill them in with "reasonable" values
movies.content_rating.fillna('UNRATED', inplace=True)

# calculate the average star rating for movies 2 hours or longer,
# and compare that with the average star rating for movies shorter than 2 hours
movies[(movies.duration >= 120)].star_rating.mean()
movies[(movies.duration < 120)].star_rating.mean()

# use a visualization to detect whether there is a relationship between star rating and duration
movies.plot(kind='scatter', x='star_rating', y='duration')

# calculate the average duration for each genre
movies.groupby(movies.genre).star_rating.mean()

'''
ADVANCED LEVEL
'''
# visualize the relationship between content rating and duration
movies['content_rating'] = movies.content_rating.map({'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4, 'X': 5})
movies.plot(kind='scatter', x='content_rating', y='duration')

# determine the top rated movie (by star rating) for each genre, answer in a nested dictionary
from collections import defaultdict
m = defaultdict(lambda: defaultdict(list))
genrelist = list(set(movies.genre))
i = 0

for x in genrelist:
    m[genrelist[i]]['top rating'] = round(movies[movies.genre == genrelist[i]].star_rating.max(),1)
    toprating = movies[movies.genre == genrelist[i]].star_rating.max()
    m[genrelist[i]]['movies'] = list(movies[(movies.genre == genrelist[i]) & (movies.star_rating == toprating)].title)
    i = i + 1
    
m.items()

# check if there are multiple movies with the same title, and if so, determine if they are the same movie
movies.title.duplicated().sum()

s = movies.sort('title')
s.index = range(1,len(s) + 1)

a = []
same = []

for index, row in s.iterrows():
    if row[1] == a:
        same.append(row[1])
    else:
        a = row[1]
        
similar = movies[movies.title.isin(same)]

print 'These are movies that have the same titles:\n',similar

print 'The number of movies with similar titles that are actually the same movies, based on the duration and the actors list, is', similar.duplicated(['duration', 'actors_list']).sum() 
similar.duplicated(['duration', 'actors_list'])

# calculate the average star rating for each genre, but only include genres with at least 10 movies
from collections import defaultdict
d = defaultdict(int)

for index, row in movies.iterrows():
    d[row[3]] = d[row[3]] + 1

greater10 = { key:value for key, value in d.items() if value > 10 }

movies[movies.genre.isin(greater10.keys())].groupby(movies.genre).star_rating.mean()
        

'''
BONUS
'''

# Figure out something "interesting" using the actors data!

