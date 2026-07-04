# Classes
class Cat:
    def __init__(self, name, age):
        self.name = name # attribute (property)
        self.age = age

    def meow(self):
        print(f"{self.name} says meow")

    # create a method to say happy birthday (increase the age)
    def happy_birthday(self):
        self.age += 1
        print(f"{self.name} is now {self.age} years old")

my_cat = Cat("ginger", 5) # creating a Cat instance
print(type(my_cat))
print(my_cat.name) # accessing an attribute
print(my_cat.age) # accessing an attribute
my_cat.meow()     # using a method

# exercise: create your_cat and make it meow
your_cat = Cat("loaf", 2)
your_cat.meow()
your_cat.happy_birthday()
your_cat.happy_birthday()

# creating a subclass
class Persian(Cat): # subclass of Cat
    def __init__(self, name, age):
        super().__init__(name, age)
        self.toy = "laser pointer"

fancy_cat = Persian("fluffy", 1)
print(fancy_cat.toy)
# print(my_cat.toy) # AttributeError