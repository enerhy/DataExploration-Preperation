# Python Stuff

import math  

# create Lists

L1 = [x**2 for x in range(10)]

L2 = []
for x in range(len(L1)):
    a = int(math.sqrt(x)*x)
    L2.append(a)
    
# Short way of creating a list
# !!!!! NO COMMAS INSIDE !!!!!
L7 = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]

#--------------------------------    

#zip function

L3 = []
for h, p in zip(L1, L2):
    a = h
    b = p
    L = [a, b]
    L3.append(L)
    
#------------------------------
    
    #!!! L2 is a pointer to an address and if we set L4=L2, L4 will point to 
    #the same address -> changing L4 results in change in L2 too 
    #the same happens the other way arount !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

L4 = L2
L4.append(99)
L2.append(0)

    #the way to make them point to different addresses and thus to a different
    #object is as follow:

L5 = list(L2)
L6 = L2[:]


#Removing an item
L5.pop(12)
del L5[11]
del L5[-1]
del L6[:2]


#Several operations 
vec = [-4, -2, 0, 2, 4]
vec = [x*2 for x in vec]
#filter the list to exclude negative numbers
vec = [x for x in vec if x>=0]
vec = [math.exp(x) for x in vec]

# using nestig functions
from math import pi
[str(round(pi, i)) for i in range(1, 6)]


# Converting a list into an Numpy array and transposing it
matrix = ([1, 2, 3],
          [4, 5, 6],
          [7, 8, 9])

import numpy as np
matrix1 = np.asarray(matrix)
matrix2 = matrix1.transpose()


#-------------------------!!! TUPLES !!!-------------------------------
t = 12345, 54321, 'hello!'
t
t[0]
#nested tuple
u = t, (1, 2, 3, 4, 5)
#!!!!!!!!!!!!!!!!!!!------Tuples are immutable-------!!!!!!!!!!!!!!!!!! 
#but they can contain mutable objects
v = ([1, 2, 3,], [4, 5, 6])


#-------------------------!!! SETS !!!-------------------------------
# a set is an unordered collection with no duplicate elements
# often used for membership testing
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
'orange' in basket

a = set('asastvfbrt')


#-------------------------!!! DICTIONARIES !!!-----------------------
# indexed by keys (string, number, tuple)
# unordered set of key: value pairs (keys are uniquie within a dict)

tel = {'jack': 4098, 'sape': 4139}
dic2 = dict(sape=4139, guido=4495, jack=4095)
tel['guido'] = 4127
del tel['sape']
tel['bobi'] = 3222

#returns a list of all the keys used
list(tel.keys())
sorted(tel.keys())
'guido' in tel

#Comprehension
di = {x: x**2 for x in (2, 4, 6)}

#Looping
for k, v in dic2.items():
    print(k, v)



#-----------------LOOPING-----------------------!!!!!!!!!!!!!!!!!!!!!!!
# Dicts
    for k, v in dic2.items():
    print(k, v)
    
# In a Sequences we can retrieve the index value pair with enumerate
    for i, v in enumerate(['tic', 'tac', 'toe']):
        print(i, v)




#-----------Classes-----------------!!!!!!!!!!!!!!!!!!!----------------!

class Employee:
    
    #class variable
    raise_amount = 1.04
    
    #this is the constructor
    #when a method is created it receives the instance as the first argument 
    #automatically #self is just how we call the instance  
    #the init method runs automatically when an instance is created

    def __init__(self, first, last, pay):
        #these are now instance variables
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        
    #that is a regular method
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
        #whithout self it wouldnt work, since by calling the method the 
        #instance is passed automatically
        
    def apply_rates(self):
        self.pay = int(self.pay * Employee.raise_amount) #!!!!class variable
                                #self.raise_amount will work as well since
                                #the instance inherits from the class
                                #if we use self we allow the method to get 
                                #a different value from the instance then
                                #the one specified as a class variable
        
    #CLASS METHOD
    @classmethod
    def set_raise_amount(cls, amount):
        cls.raise_amount = amount
    
    #alternative constructor
    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)
    
    #STATIC METHODS - dont depend on instance ot a class variable
    #they have a logical connection to the class
    #we use it if we dont access an instance or the class in the method
    @staticmethod
    def is_workday(day):
        if day.is_workday() < 5:
            return False
        return True
    
    
    
#----------
        
        
#Namespaces
print(Employee.__dict__)
print(emp_2.__dict__)
        
emp_2 = Employee('Todor', 'Jivkov', 5000)
print(emp_2.email)
print(emp_2.fulname())
#this is the same as 
print(Employee.fulname(emp_2))

#Using the class variables
print(emp_2.raise_amount)
print(Employee.raise_amount)

Employee.raise_amount = 1.04
emp_2.raise_amount = 1.09 #here we create a rais

print(emp_2.raise_amount)
print(emp_1.raise_amount)
print(Employee.raise_amount)

emp_1 = Employee('Gosho', 'Goshev', 10000)
print(emp_1)

#instance variables
emp_1.first = 'Angel'


#----------Class Methods
Employee.set_raise_amount(1.10)
 
#alternative constructor
emp_str_1 = 'John-Doe-7000'
new_emp_1 = Employee.from_string(emp_str_1)


import datetime
my_date = datetime.date(2016, 7, 10)
print(my_date)
print(Employee.is_workday(my_date))


#INHERITANCE---------------INHERITANCE---------------INHERITANCE-------------

class Developer(Employee):
    #class variable of the Subclass
    raise_amountt = 1.10
 
    def __init__(self, first, last, pay, prog_lang):    
        super().__init__(first, last, pay) #the Superclass handles this arguments
        self.prog_lang = prog_lang


class Manager(Employee):
    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees
            
    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
            
    def remove_emp(self, emp):
        if emp  in self.employees:
            self.employees.remove(emp)

    def print_emp(self):
        for emp in self.employees:
            print('-->', emp.fullname())

----
print(help(Developer))

dev_1 = Developer('Angel', 'Spasov', 50000, 'Python')
dev_2 = Developer('Magdalena', 'Spasova', 80000, 'C++')

print(dev_1.first, dev_1.prog_lang)
print(dev_1.fullname())
    
mgr_1 = Manager('Bacho', 'Kiro', 90000, [dev_1])
print(mgr_1.email)

mgr_1.print_emp()
mgr_1.add_emp(dev_2)
mgr_1.print_emp() 
    
    
print(isinstance(mgr_1, Manager))
print(isinstance(mgr_1, Developer))   
print(issubclass(Developer, Manager))   
print(issubclass(Developer, Employee))   


#-----PROPERTY DECORATOR--->>>>--GETTER-----SETTER------DELETERS--------!!!!
# allow us to create a method but use/access it as an attribute

class Employee2:
    
    def __init__(self, first, last):
    #these are now instance variables
        self.first = first
        self.last = last
        
        # This is substituted with the property decorator: 
        # self.email = first + '.' + last + '@company.com'
        
    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)
    
    #this property is needed for the setter
    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last
    
    #in order to delete an attribute
    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None
        

emp_4 = Employee2('John', 'Smith')

emp_4.first = 'Jim'

emp_4.fullname = 'Christiano Ronaldo'

print(emp_4.first)
print(emp_4.email)
print(emp_4.fullname)







