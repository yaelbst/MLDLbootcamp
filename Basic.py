##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Feb 28 13:24:35 2019
#
#@author: yael
#"""
#
#Part 1 – Python basic # The lines that start with a hash (#) are comments
## They are for you to read and are ignored by Python
## When you see 'GO!', save and run the file to see the output
## When you see a line starting with # follow the instructions
## Some lines are python code with a # in front
## This means they're commented out - remove the # to uncomment # Do one challenge at a time, save and run after each one!
## 1.1. This is the print statement 
print("Hello world")
## GO!
## 1.2. This is a variable
message = "Level Two"
## Add a line below to print this variable # GO!
print(message)
## 1.3. The variable above is called a string
## You can use single or double quotes (but must close them)
## You can ask Python what type a variable is. Try uncommenting the next line: 
print(type(message))
## GO!
## 1.4. Another type of variable is an integer (a whole number) 
a = 123
b = 654
c= a + b
## Try printing the value of c below to see the answer # GO!
print(c)

## 1.6. Variables keep their value until you change it 
a = 100
print(a) # think - should this be 123 or 100? 
c = 50
# 
print(c) # think - should this be 50 or 777?
d = 10 + a - c
print(d) # think - what should this be now?
## GO!
## 1.7. You can also use '+' to add together two strings
greeting = 'Hi '
name = 'Yael' # enter your name in this string
message = greeting + name 
print(message)
## GO!
## 1.8. Try adding a number and a string together and you get an error: 
age = 41 # enter your age here (as a number)
##print(name + ' is ' + age + ' years old')
## GO!
## See the error? You can't mix types like that.
## But see how it tells you which line was the error? # Now comment out that line so there is no error
## 1.9. We can convert numbers to strings like this: #
print(name + ' is ' + str(age) + ' years old')
## GO!
## No error this time, I hope?
## Or we could just make sure we enter it as a string: # 
age = "41"# enter your age here, as a string
print(name + ' is ' + age + ' years old')
## GO!
#
## No error this time, I hope?
## 1.10. Another variable type is called a boolean # This means either True or False
raspberry_pi_is_fun = True 
raspberry_pi_is_expensive = False
## We can also compare two variables using ==
bobs_age = 15
your_age = 41# fill in your age
print(your_age == bobs_age) # this prints either True or False
## GO!
## 1.11. We can use less than and greater than too - these are < and > # 
bob_is_older = bobs_age > your_age
print(bob_is_older) # do you expect True or False?
## GO!
## 1.12. We can ask questions before printing with an if statement
money = 500 
phone_cost = 240 
tablet_cost = 200
total_cost = phone_cost + tablet_cost 
can_afford_both = money > total_cost
if can_afford_both:
  message = "You have enough money for both"
else:
  message = "You can't afford both devices"
print(message) # what do you expect to see here? # GO!
## Now change the value of tablet_cost to 260 and run it again # What should the message be this time?
tablet_cost = 260
total_cost = phone_cost + tablet_cost 
can_afford_both = money >= total_cost
if can_afford_both:
  message = "You have enough money for both"
else:
  message = "You can't afford both devices"
print(message)
## GO!
## Is this right? You might need to change the comparison operator to >= # This means 'greater than or equal to'
raspberry_pi = 25
pies = 3 * raspberry_pi
total_cost = total_cost + pies
if total_cost <= money:
  message = "You have enough money for 3 raspberry pies as well"
else:
  message = "You can't afford 3 raspberry pies"
print(message) # what do you expect to see here?
## GO!
## 1.13. You can keep many items in a type of variable called a list
colours = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
## You can check whether a colour is in the list
print('Black' in colours) # Prints True or False
## GO!
## You can add to the list with append
colours.append('Black') 
colours.append('White')
print('Black' in colours) # Should this be different now? # GO!
## You can add a list to a list with extend
more_colours = ['Gray', 'Navy', 'Pink'] 
colours.extend(more_colours)
print(colours)
## Try printing the list to see what's in it
## GO!
## 1.14. You can add two lists together in to a new list using +
primary_colours = ['Red', 'Blue', 'Yellow'] 
secondary_colours = ['Purple', 'Orange', 'Green']
main_colours = primary_colours + secondary_colours
## Try printing main_colours
print(main_colours)
## 1.15. You can find how many there are by using 
print(len(main_colours))
## How many colours are there in main_colours?
## GO!
all_colours = colours + main_colours
## How many colours are there in all_colours?
## Do it here. Try to think what you expect before you run it
print(len(all_colours))
## GO!
## Did you get what you expected? If not, why not?
## 1.16. You can make sure you don't have duplicates by adding to a set
even_numbers = [2, 4, 6, 8, 10, 12] 
multiples_of_three = [3, 6, 9, 12]
numbers = even_numbers + multiples_of_three 
print(numbers, len(numbers))
numbers_set = set(numbers)
print(numbers_set, len(numbers_set))
## GO!
colour_set = set(all_colours)
## How many colours do you expect to be in this time? 
# Do you expect the same or not? Think about it first
print(colour_set, len(colour_set))
## 1.17. You can use a loop to look over all the items in a list
#
my_class = ['Sarah', 'Bob', 'Jim', 'Tom', 'Lucy', 'Sophie', 'Liz', 'Ed']
## Below is a multi-line comment
## Delete the ''' from before and after to uncomment the block

for student in my_class:
  print(student) 
## Add all the names of people in your group to this list
## Remember the difference between append and extend. You can use either.
my_class.append("Liora")
print(my_class)
my_class.extend(["Shlomit", "Yael"])
print(my_class)
## Now write a loop to print a number (starting from 1) before each name
for n in range(len(my_class)):
    print(str(n+1) +' '+ my_class[n])
## 1.18. You can split up a string by index 
full_name = 'Dominic Adrian Smith'
first_letter = full_name[0]
last_letter = full_name[19]
first_three = full_name[:3] # [0:3 also works]
last_three = full_name[-3:] # [17:] and [17:20] also work middle = full_name[8:14]
## Try printing these, and try to make a word out of the individual letters 
print(first_letter)
print(last_letter)
print(first_three)
print(last_three)
# 1.19. You can also split the string on a specific character
my_sentence = "Hello, my name is Fred" 
parts = my_sentence.split(',')
print(parts)
print(type(parts)) # What type is this variable? What can you do with it?
## GO!
my_long_sentence = "This is a very very very very very very long sentence" 
# Now split the sentence and use this to print out the number of words
my_words = my_long_sentence.split(' ')
print(len(my_words))
## GO! (Clues below if you're stuck)
#
## Clue: Which character do you split on to separate words? # Clue: What type is the split variable?
## Clue: What can you do to count these?
##1.20. You can group data together in a tuple
person = ('Bobby', 26)
print(person[0] + ' is ' + str(person[1]) + ' years old') 
# GO!

students = [('Dave', 12), ('Sophia', 13), ('Sam', 12), ('Kate', 11), ('Daniel', 10)]
## Now write a loop to print each of the students' names and age
for name, age in students:
    print(name,age)
## GO!
## 1.21. Tuples can be any length. The above examples are 2-tuples.
## Try making a list of students with (name, age, favourite subject and sport)
new_students = [('Ido',23, 'machine learning','tennis'),('Sofia', 54, 'deep learning', 'swimming'),('Lauren', 36, 'classification', 'golf')]
## Now loop over them printing each one out
for name, age, subject, sport in new_students:
    print(name,age,subject,sport)
## Now pick a number (in the students' age range)
for name, age, subject, sport in new_students:
    if age <= 36:
      print(name,age,subject,sport)
## Make the loop only print the students older than that number
## GO!
## 22. Another useful data structure is a dictionary
## Dictionaries contain key-value pairs like an address book maps name # to number
addresses = {'Lauren': '0161 5673 890', 'Amy': '0115 8901 165', 'Daniel': '0114 2290 542','Emergency': '999' }
## You access dictionary elements by looking them up with the key:
print(addresses['Amy'])
## You can check if a key or value exists in a given dictionary:
print('David' in addresses) # [False]
print('Daniel' in addresses) # [True]
print('999' in addresses) # [False]
print('999' in addresses.values()) # [True]
print(999 in addresses.values()) # [False]
## GO!
## Note that 999 was entered in to the dictionary as a string, not an integer
## Think: what would happen if phone numbers were stored as integers?
## Try changing Amy's phone number to a new number
addresses['Amy'] = '0115 236 359'
print(addresses['Amy'])
## GO!
#Delete Daniel from the dictinary
print('Daniel' in addresses) # [True]
del addresses['Daniel']
print('Daniel' in addresses) # [False]
## GO!
## You can also loop over a dictionary and access its contents:
#'''
for name in addresses:
  print(name, addresses[name])
## GO!
#
## 1.23. A final challenge using the skills you've learned:
## What is the sum of all the digits in all the numbers from 1 to 1000?
sum = 0
for n in range(1000):
    strn = str(n)
    for ch in strn:
        sum += int(ch)
print(sum)
## GO!
## Clue: range(10) => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Clue: str(87) => '87'
## Clue: int('9') => 9
#1.24. Define a function `max()` that takes two numbers as arguments and returns the largest of them. Use the if-then-else construct available in Python. (It is true that Python has the `max()` function built in, but writing it yourself is nevertheless a good exercise ).
def max(n,m):
  if n > m:
      return n
  elif m > n:
      return m
  else:
      print("The two numbers are equal")

print(max(2,3))
#1.25. Define a function `max_of_three()` that takes three numbers as arguments and returns the largest of them.

def max_of_three(x,y,z):
    if (x > y) and (x>z):
        return x
    elif (y>x) and (y>z):
        return y
    elif (z>x) and (z>y):
        return z
    
print(max_of_three(4,12,8))
#1.26. Define a function that computes the length of a given list or string.
# ( It is true that Python has the `len()` function built in, but writing it yourself is nevertheless a good exercise ).
sentence = "My sentence is a list in itself"

def sentence_len(my_sentence):
 count = 0
 for ch in sentence:
    count += 1
 return count

print(sentence_len(sentence))

#1.27. Write a function that takes a character ( i.e. a string of length 1 ) and returns `True` if it is a vowel, `False` otherwise.
# i use a tuple instead of list to try it out and use less space

def is_vowel(ch):
    vowels = ('a','e','i','o','u','y')
    if ch in vowels:
        return True
    else:
        return False

print(is_vowel("y"))
print(is_vowel("r"))
#1.28. Write a function `translate()` that will translate a text into "rövarspråket" (Swedish for "robber's language"). That is, double every consonant and place an occurrence of "o" in between. For example, `translate("this is fun")` should return the string `"tothohisos isos fofunon".`
def turn_Swedish(text):
   vowels = ('a','e','i','o','u','y')
   new_text = ''
   for ch in text:
       if (ch not in vowels) and (ch != ' '):
           new_text += ch+'o'+ch
       else:
           new_text += ch
   return new_text

print(turn_Swedish("This is fun"))
#Part 2 - advanced
#2.1
#Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5,
#between 2000 and 3200 (both included).
#The numbers obtained should be printed in a comma-separated sequence on a single line.
#Hints:
#Consider use range(#begin, #end) method

relevant_numbers= []
for n in range(2000,3201):
    if (n%7 == 0) and (n%5!=0):
        relevant_numbers.append(n)
print(relevant_numbers)
    
#2.2
#Write a program which can compute the factorial of a given numbers.
#The results should be printed in a comma-separated sequence on a single line.
def factorial(my_num):
   product = 1
   intermediaries = []
   for n in range(1, my_num+1):
       product = product*n
       intermediaries.append(product)
   print(intermediaries)
   return product

print(factorial(8))
#Suppose the following input is supplied to the program:
#8
#Then, the output should be:
#40320
#Hints:
#In case of input data being supplied to the question, it should be assumed to be a console input.
#2.3
#With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number between 1 and n (both included). and then the program should print the dictionary.
def dic_square(n):
    dic_square = {}
    for i in range(1,n+1):
        dic_square.update({i:i*i})
    return dic_square

print(dic_square(8))
#Suppose the following input is supplied to the program:
#8
#Then, the output should be:
#{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}
#Hints:
#In case of input data being supplied to the question, it should be assumed to be a console input.
#Consider use dict()
#2.4

#I put this in comment so it doesn't stop me in the following exercises
#str_num_sequence = input("Enter a sequence of integers separated by a comma")

#list_sequence = str_num_sequence.split(',')
#tu_sequence = tuple(list_sequence)
#big_list = [list_sequence, tu_sequence]

#print(str(big_list[0])+' '+str(big_list[1]))



#Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number. Suppose the following input is supplied to the program:
#34,67,55,33,12,98
#Then, the output should be: ['34', '67', '55', '33', '12', '98'] ('34', '67', '55', '33', '12', '98')
#Hints:
#In case of input data being supplied to the question, it should be assumed to be a console input.
#tuple() method can convert list to tuple
#2.5
#Write a method which can calculate a square value of a number
def compute_square(n):
    return n*n

print(compute_square(4))
#2.6
#
#Python has many built-in functions, and if you do not know how to use it, you can read document online or find some books. But Python has a built-in document function for every built-in functions.
#Please write a program to print some Python built-in functions documents, such as abs(), int(), raw_input()
#And add document for your own function
#Hints:
#The built-in document method is __doc__
print(abs.__doc__)
#2.7.
#class
#Answer the following questions and write the following classes
#1. What is a class? an object
#2. What is an instance? one occurence of an object
#3. What is encapsulation?
#4. What is inheritance? get proprieties of an object from another one 
#5. What is polymorphism?
#6. Triangle Class:
#a. Create a class, Triangle. Its __init__() method should take self,
#angle1, angle2, and   as arguments. Make sure to set these
#appropriately in the body of the __init__()method.
#b. Create a variable named number_of_sides and set it equal to 3.
#c. Create a method named check_angles. The sum of a triangle's
#three angles is It should return True if the sum of self.angle1, self.angle2, and self.angle3 is equal 180, and False otherwise.
#d. Create a variable named my_triangle and set it equal to a new
#instance of your Triangle class. Pass it three angles that sum to 180
#(e.g. 90, 30, 60).
#e. Print out and print out
class Triangle:
    
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
    
    number_of_sides = 3
    
    def check_angles(self):
        if self.angle1+self.angle2+self.angle3 == 180:
            return True
        else:
            return False
    
my_triangle=Triangle(90,30,60)
print(my_triangle.number_of_sides)
print(my_triangle.check_angles())
#.
#7. Songs class:
#a. Define a class called Songs, it will show the lyrics of a song. Its
#__init__() method should have two arguments:   anf   . is
#        my_triangle.number_of_sides
# my_triangle.check_angles()
#   a list. Inside your class create a method called
#prints each element of     on his own line. Define a varible:
#b. , ,
#that
#  "Have a sunshine on you,"
#c. Call the
class Song:
    
    def __init__(self, lyrics):
        self.lyrics = lyrics
    
    def sing_me_a_song(self):
        for l in self.lyrics:
            print(l)
            
happy_bday = Song(["May god bless you, ", "Have a sunshine on you!", "Happy b-day to you!"])
happy_bday.sing_me_a_song()

class Lunch:
    
    def __init__(self, menu):
        self.menu = menu
    
    def menu_price(self):
        if self.menu == "menu 1":
            print("Your choice ", self.menu ," - Price 12.00")
        elif self.menu == "menu 2":
            print ("Your choice ", self.menu, "Price 13.40")
        else:
            print("Error in menu")
    
Paul=Lunch("menu 1")   
Paul.menu_price()
    
#8. Define a class called   . Its __init__() method should have two
#method on this variable
#self
#lyrics
#Lyrics
# happy_bday = Song(["May god bless you, "
#lyrics
#sing_me_a_song
#  "Happy Birthday to you !"])
# sing_me_song
# Lunch
#angle3
# arguments:selfanf   .Where menu is a string. Add a method called .It will involve a ifstatement:
#if "menu 1" print "Your choice:", menu, "Price 12.00", if "menu 2" print "Your choice:", menu, "Price 13.40", else print "Error in menu". 
#To check if it works define: Paul=Lunch("menu 1") and call Paul.menu_price().
#menu
#  menu_price
#  
# Define a Point3D class that inherits from object Inside the Point3D class,
#   define an
#__init__()
class Point3D:
    
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
    def __repr__(self):
        return "(%d, %d, %d)" % (self.x, self.y, self.z)

my_point = Point3D(1,2,3)
print(my_point)
        
#function that accepts self, x, y, and z, and assigns
#       these numbers to the member variables
#self.x,
#self.y,
#self.z
#. Define a
#   __repr__()
#method that returns
#"(%d, %d, %d)" % (self.x, self.y, self.z)
# This tells Python to represent this object in the following format: (x, y, z).
#   Outside the class definition, create a variable named
#my_point
#containing
# a new instance of Point3D with x=1, y=2, and z=3. Finally, print
# my_point
#9.
# .
# .
#You can find the solutions to these exercises in the link:
#2.8.
# https://erlerobotics.gitbooks.io/erle-robotics-learning-python-gitbook-
#  free/classes/exercisesclasses.html

#  List comprehension - a quick reminder
#  1. Use dictionary comprehension to create a dictionary of numbers 1- 100 to
#their squares (i.e. {1:1, 2:4, 3:9 ...}
import time
import timeit

time1 = time.time()
time1i = timeit.timeit()
print("time1", time1)
print("timeit", time1i)
dcarre = {}
for x in range(1,101):
    dcarre.update({x : x*x})
time2 = time.time()
print("time2", time2)

dict_square = {x: x*x for x in range(1,101)}
time3 = time.time()
print("time3", time3)

try:
    print(dict_square)
except:
    print(Exception)

time4 = time2 - time1
time5 = time3 - time2
print(time4)
print(time5)

#2. Use list comprehension to create a list of prime numbers in the range 1-
#100
list_prime = [x for x in range(1, 101)
     if all(x % y != 0 for y in range(2, x))]
print(list_prime)
# http://www.secnetix.de/olli/Python/list_comprehensions.hawk
#  2.9.
# Files
#Use open, read, write, close which are explained in these tutorials: https://www.tutorialspoint.com/python/python_files_io.htm https://www.guru99.com/reading-and-writing-files-in-python.html to:

my_file = open("my_file.txt", "w")
my_file.write("Hello world!")
my_file.close()

# Open a file
mf = open("my_file.txt", "r")
strmessage = mf.read()
print ("Read String is : ", strmessage)
# Close opend file
mf.close()
#    1. Write “Hello world” to a file
#2. Read the file back into python.

#3. Write the square roots of the numbers 1-100 into the file, each in new
#line.
import math 
f = open("my_file.txt", "a")
for n in range(1, 101):
    f.write(str(math.sqrt(n)) + "\n")
f.close()

with open('Eigenface.txt', 'w') as fl:
     fl.write('Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition.[1] The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification.[2] The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set.') 
fl.close()

with open('Eigenface.txt', 'r') as fe:  
    mess = fe.readlines()
    print("Eigenface:" + mess[0])
fe.close()

eigen= ""
with open('Eigenface.txt', 'r') as fi:  
    list_mess = fi.read().split(' ')
    for line in list_mess:
      print(line)
    one_line = eigen.join(list_mess)
    print(one_line)
fi.close()

print("2.11 optional Start----")
#2.11 optional
para = 'Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition.[1] The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification.[2] The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set.'
phrases = para.split('.')
for phrase in phrases:
    new_phrase= phrase.split(' ')
    #print(new_phrase[::-1])
    rev_phrase = ""
    for word in new_phrase[::-1]:
        rev_phrase += " "+word
    print(rev_phrase)
    print("\n")

print("211 optional END")
# 2.10
# With
#Read about with in one of the following links: https://stackoverflow.com/questions/1369526/what-is-the-python-keyword- with-used-for
#   http://effbot.org/zone/python-with-statement.htm
#  1. Open a file using with and write the first paragraph from (you don’t need
#to read the webpage, just copy paste it into code)
#https://en.wikipedia.org/wiki/Eigenface
#2. Open the file using with and read the contents using readlines.
#  2.11.
# Strings
#  1. Split the paragraph to a list of words by spaces (hint: split())
#
# 2. Join the words back into a long string using “join”. (hint: join())
#* (optional) Create a paragraph with the word order reversed in each
#sentence (but keep the order of the sentences)

#3. write a function accepting two numbers and printing “the sum of __ and
#___ is ___”.
#2.12
#2.13
import time

def sum_of_two(a,b):
    the_sum = a+b
    str_sum = str(the_sum)
    return "The sum of " +str(a)+ " and "+str(b)+ " is "+ str_sum


print(sum_of_two(3,4))


def sec_sum_of_two(a,b):
 #   try:
      if (a<=100) and (b<=100):
          return "The sum of {} and {} is {}".format(a,b,a+b)
      else:
          raise Exception("a and/or b are/is too big")
 #   except:
 #     print("TypeError: you entered a string instead of a number")
      


print(sec_sum_of_two(6,98))

import datetime
the_date = datetime.datetime.now()

print(the_date)

today = datetime.date.today()
shilshom = today - datetime.timedelta(days = 2)

print(shilshom)

in12hours = the_date + datetime.timedelta(hours = 12)
print(in12hours)


print(time.time())
# Do this twice using one string formats learned in class each time. Hint: https://pyformat.info/
#2.12.
# Datetime, time, timeit
#Read the current datetime to a variable usint datetime.datetime.no see: https://docs.python.org/3/library/datetime.html
#10. Print the date of the day before yesterday, and the datetime in 12 hours from now
#   Use time library to time library to time operations by storing time.time() before and after running an operation https://www.tutorialspoint.com/python/time_time.htm
#  1. Time one of the comprehension exercises you’ve performed and compare to a simple for implementation, to determine which one is faster
#  1. Now use timeit library to time your list comprehension operation, see:
# https://docs.python.org/2/library/timeit.html
#  2.13.
# Exceptions
#Read about exceptions from: http://www.pythonforbeginners.com/error-handling/exception-handling-in- python
#    1. Enhance the function you wrote which reads numbers (Strings 3.)
# a. In case of a string input writing an error message instead of raising
#an error
#b. Raise an error in case one of the input numbers is > 100
#c. Create a code which tries to read from the dictionary of squares
#created
# 2.14.
#Logger
#Read about loggers from the example: https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/ Create a logger in any of your programs and log various messages to a file using a FileHandler, and to a the console using StreamHandler
#*Optional : try setting different levels (e.g. debug for console and info for file) and different formats and try different levels of logs in your code.
#  


#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

# create a file handler
#handler = logging.FileHandler('hello.log')
#
# create console handler
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)

# create a logging format
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
#ch.setFormatter(formatter)

# add the handlers to the logger
#logger.addHandler(handler)
#logger.addHandler(ch)

#logger.info('Hello baby')

list_prime = [x for x in range(1, 32)
     if all(x % y != 0 for y in range(2, x))]
#logger.info("prime numbers: done")
#2.15
#Regular expressions (re)
#You can read about regex from the following sources:
#Python documentation - https://docs.python.org/3/library/re.html
#General links about RegEx - http://www.regular-expressions.info/tutorial.html And - https://www.icewarp.com/support/online_help/203030104.htm
#(The last is more brief and practical)
#Try to understand the difference between search and match (and compile), and how to match the strings exactly.
#1. To get the hang of it, try to do a few of the the exercises in: https://regexone.com/
#2. To get a sense of regex in python, take a look at the first few exercises in: http://www.w3resource.com/python-exercises/re/
#3*. Optional: Write a regular expression for identifying email addresses, find and compare to regular expression found online.
#2.16.
#my_file = open("combine.txt", "w")
#my_file.write("5 100 - 17")
#my_file.close()
import re
import logging

def is_allowed_specific_char(string):
    charRe = re.compile(r'[^a-zA-Z0-9.+*/-]')
    string = charRe.search(string)
    return not bool(string)


my_logger = logging.getLogger(__name__)
my_logger.setLevel(logging.WARNING)

# create a file handler
my_handler = logging.FileHandler('range_log.log')
my_handler.setLevel(logging.WARNING)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

# create a logging format
my_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
my_handler.setFormatter(my_formatter)
ch.setFormatter(my_formatter)

# add the handlers to the logger
my_logger.addHandler(my_handler)
my_logger.addHandler(ch)
    
def range_calculator(filename):

    
    time1 = time.time()
    the_file = open(filename,"r")
    fl =the_file.readlines()
    the_file.close()
    logging.warning('Original file read')
    logging.warning('Will show in CONSOLE')
    sec_file = open("results.txt", "w")
    ops = {"+": (lambda x,y: x+y), "-": (lambda x,y: x-y), "*": (lambda x,y: x*y), "/":(lambda x,y: x/y), "**": (lambda x,y: x**y)}
    for l in fl:
          print(l)
          ll = l.split(' ')
          endlist = float(ll[3].replace('\n',''))
          basic_list = list(range(int(ll[0]), int(ll[1])))
          for el in basic_list:
             if is_allowed_specific_char(str(el)):
               comp_el = ops[ll[2]](el,endlist)
               sec_file.write(str(round(comp_el, 2))+",")
             else:
               raise Exception("The original file contains at least one forbidden charcater")  
          sec_file.write("\n")
    #endel = ops[ll[2]] (int(ll[1]), endlist)
    sec_file.close() 
         
    time2 = time.time()
    op_duration = time2 - time1
    logging.warning("The whole process took" + str(op_duration))

range_calculator("combine.txt")
logging.warning("Hi CONSOLE")
#Combine everything!
#Write a simple “range calculator” function which reads a file and writes the result into another file. You should log the operations, and how much time they took and handle problematic inputs (strings, wrong characters, etc.). Validate the input using regex.
#The file will contain in each line two numbers denoting a range of numbers and they can only be integers, an operation denoted by a sign (+,*,-,/,**) and a third number which will be applied with the operation for each of the numbers in the range. Note all numbers and signs are separated by blanks (please don’t use .csv readers and use plain readers):
#5 100 - 17
#18 25 * 2.784
#(First line: subtract 17 from all numbers between 5 and 100, Second line: multiply by 2.784 all numbers between 18 and 25)
#The result of each range should be written into one line, separated with commas and with the precision of two digits after the decimal point.
#2.17. os
#Os module provides a portable way of using operating system dependent functionality you can read about os in this link. https://docs.python.org/3/library/os.html
#1. print absolute path on your system
#2. print files and directories in the current directory on your system
import os
print(os.path.abspath("myfile.txt")) 
print(os.listdir("."))     
# 2.18.
# glob.
import glob
#below in comment because it's too much print out in console, bothers me to see the rest
print(glob.glob(r"/Users/yael/Downloads/*.pdf"))
# The glob module finds all the pathnames matching a specified pattern
# according to the rules used by the Unix shell.
# You can read about it in the next link.
#https://docs.python.org/3/library/glob.html
#Find in your download directory all the files that have extension pdf (.pdf)
#2.19.
#enumerate
#http://book.pythontips.com/en/latest/enumerate.html
#Write a list with some values, use for and enumerate to print in each iteration the value and his index
L = ["blue","white","yellow","orange","pink"]

for i,el in enumerate(L):
    print("index :" + str(i) +" element :"+ el)

#2.20.
#Threads
#1. Write two functions, one that writes “SVD” article from wikipedia to a file and one that calculates the sum of two numbers. Run the two functions
#simultaneously using threads.
from urllib.request import urlopen
from threading import Thread, Lock

def write_SVDarticle_from_url():
    link = "https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html"
    f = urlopen(link)
    my_file = open("svd_content.txt", "w")
    content = f.read()
    my_file.write(str(content))
    my_file.close()

def sum_of_2_numbers(a,b):
    print(a+b)
    return a+b

t1 = Thread(target=write_SVDarticle_from_url)
t2 = Thread(target=sum_of_2_numbers, args=(3,4))

t1.start()
t2.start()
#2. Use Threading.Lock() in order to acquire and release mutexes so the two
#functions don’t run simultaneously.
#Hint: http://www.python-course.eu/threads.php

# in the case that the first function doesn't have any argument, it didn't work to pass a mutex. 
# I then had to add a "fake" argument p, p=1, so the solution worked
#p=1
# So i changed the solution to pass the link as an argument



#open 
#w opens for writing will truncate
#a+	open file for reading and writing (appends to end)

def write_article_from_url_mutex(link,mutex):
    mutex.acquire()
    response = urlopen(link)
    my_file = open("artm_content.txt", "w")
    content = response.read()
    my_file.write(str(content))
    my_file.close()
    response.close()
    mutex.release()
    
def sum_of_2_numbers(a,b, mutex):
    mutex.acquire()
    print(a+b)
    mutex.release()
    return a+b

mutex = Lock()
link = "https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html"
th1 = Thread(target=write_article_from_url_mutex, args=(link,mutex))
th1.start()
th2 = Thread(target=sum_of_2_numbers, args=(3,4,mutex))
th2.start()
#2.21
#Random https://docs.python.org/3/library/random.html
#Create a list with 10 values (string)
#Take only one value from the list from a random place
#Split the list to two lists in different sizes each list contain values from the previous list but in random order.
import random

new_list = ['aa','bb','cc','dd','ee','ff','gg','hh','ii','jj']
new_new_list = random.shuffle(new_list)
print(new_new_list)
ran_index = random.randint(1,9)
print('ran_index is: '+str(ran_index))
print(new_list[ran_index])
small_list1 = new_list[0:ran_index]
small_list2 = new_list[ran_index:]
print(small_list1, small_list2)
##      