'''
Python Homework with Chipotle data
https://github.com/TheUpshot/chipotle
'''

'''
BASIC LEVEL
PART 1: Read in the data with csv.reader() and store it in a list of lists called 'data'.
Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.
      https://docs.python.org/2/library/csv.html
'''

import csv
with open('chipotle.tsv', 'rU') as tsvfile:
    data = [row for row in csv.reader(tsvfile, delimiter="\t")]
    

'''
BASIC LEVEL
PART 2: Separate the header and data into two different lists.
'''
header = data[0]
data = data[1:]


'''
INTERMEDIATE LEVEL
PART 3: Calculate the average price of an order.
Hint: Examine the data to see if the 'quantity' column is relevant to this calculation.
Hint: Think carefully about the simplest way to do this!
'''
priceperorder = []
orderlist = [0]
i = 0
price = 0
for row in data:
    orderid = row[0]
    p = ''.join(e for e in row[4] if e.isalnum())
    itemprice = float(p)/100
    
    if orderid > orderlist[i]:
        priceperorder.append(round(price, 2))        
        price = itemprice
       
    else:
        price = price + itemprice        
        
    orderlist.append(orderid)
    
    i = i+1

avg = sum(priceperorder[1:])/len(priceperorder[1:])

print 'The average price per order is $',round(avg,2)
    

'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.
'''
soda = []
for row in data:
    if row[2] =='Canned Soda':
        soda.append(row[3])
    else:
        next

soda = set(soda)

print 'The list of unique sodas is the following:', soda

soft = []
for row in data:
    if row[2] =='Canned Soft Drink':
        soft.append(row[3])
    else:
        next

soft = set(soft)

print 'The list of unique soft drinks is the following:', soft

'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!
'''
toplist= []
j = 0
for row in data:
    item = row[2]
    if item.find("Burrito") == -1:
        next
    else:
        toppings = len(row[3].split(','))
        toplist.append(toppings)

avgtoppings = sum(toplist)/len(toplist)

print 'The average number of toppings per burrito is',avgtoppings

'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.
'''
chiplist = []
for row in data:
    if row[2].find("Chips") == -1:
        next
    else:
        chiplist.append(row[2])

chiplist = set(chiplist)

from collections import defaultdict
d = defaultdict(list, zip(chiplist, [0,0,0,0,0,0,0,0,0,0,0]))

for row in data:
    item = row[2]
    n = int(row[1])
    if item.find("Chips") == -1:
        next
    else:
        d[item] = d[item] + n


print 'The following is a dictionary representing chip orders and the respective total number of orders:', d