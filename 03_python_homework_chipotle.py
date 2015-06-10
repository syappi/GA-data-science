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
        priceperorder.append(price)        
        price = itemprice
       
    else:
        price = price + itemprice        
        
        
        
    orderlist.append(orderid)
    
    i = i+1

avg = sum(priceperorder[1:])/len(priceperorder[1:])

print 'The average price per order is $',avg

'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.
'''



'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!
'''



'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.
'''