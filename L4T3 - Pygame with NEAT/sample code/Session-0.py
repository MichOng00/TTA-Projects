'''A brief introduction to classes in Python.'''
# rmb need to install VS Code, python extension
# todo: define class, object, method, attribute
# not OOP

# version 1 - absolute minimal example
class Dog:
    def __init__(self): # constructor
        pass

dog = Dog()
print(type(dog))

############################################
# version 2
class Dog:
    '''A simple class to represent a dog.'''
    
    def __init__(self, name, age):
        '''Initialize the dog.'''
        self.name = name  # attribute
        self.age = age
        self.sitting = False

    def bark(self):
        '''Make the dog bark.'''
        print(f"{self.name} says Woof!")

    def sit(self):
        '''Make the dog sit.'''
        self.sitting = True
        print(f"{self.name} is now sitting.")

    def stand(self):
        '''Make the dog stand up.'''
        self.sitting = False
        print(f"{self.name} is now standing.")

    def happy_birthday(self):
        '''Celebrate the dog's birthday by increasing its age.'''
        self.age += 1
        print(f"Happy Birthday {self.name}! Now you are {self.age} years old.")

# Create an instance of the Dog class
my_dog = Dog("Buddy", 3)
print(type(my_dog))  # Output: <class '__main__.Dog'>
print(f"My dog's name is {my_dog.name} and he is {my_dog.age} years old.")
my_dog.bark()  # Output: Buddy says Woof!

# Create another instance of the Dog class
another_dog = Dog("Max", 5)
print(type(another_dog))
print(f"My dog's name is {another_dog.name} and he is {another_dog.age} years old.")
another_dog.bark()
another_dog.sit() 
another_dog.happy_birthday()
another_dog.happy_birthday()
print(another_dog.age)

####################################################
# version 3
class Dog:
    '''A simple class to represent a dog.'''

    def __init__(self, name, age):
        '''Initialize the dog.'''
        self.name = name
        self.age = age
        self.sitting = False
        self.energy = 100

    def bark(self):
        print(f"{self.name} says Woof!")

    def sit(self):
        self.sitting = True
        print(f"{self.name} is now sitting.")

    def stand(self):
        self.sitting = False
        print(f"{self.name} is now standing.")

    def happy_birthday(self):
        self.age += 1
        print(f"Happy Birthday {self.name}! Now you are {self.age} years old.")

    def sleep(self):
        self.energy = 100
        print(f"{self.name} is sleeping... Energy restored to {self.energy}.")

    def play(self):
        if self.energy >= 10:
            self.energy -= 10
            print(f"{self.name} is playing! Energy is now {self.energy}.")
        else:
            print(f"{self.name} is too tired to play.")

    def describe(self):
        print(f"{self.name} is {self.age} years old. Sitting: {self.sitting}. Energy: {self.energy}.")

####################################################
# user interactions

this_dog = Dog("Buddy", 3)
while True:
    action = input("What do you want to do? (bark, sit, stand, birthday, sleep, play, exit): ").strip().lower()
    
    if action == "describe":
        this_dog.describe()
    elif action == "bark":
        this_dog.bark()
    elif action == "sit":
        this_dog.sit()
    elif action == "stand":
        this_dog.stand()
    elif action == "birthday":
        this_dog.happy_birthday()
    elif action == "sleep":
        this_dog.sleep()
    elif action == "play":
        this_dog.play()
    elif action == "exit":
        print("Exiting the program.")
        break
    else:
        print("Invalid action. Please try again.")