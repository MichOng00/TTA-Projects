'''Builds on session 0 to add class inheritance.'''
# todo: check actual session 0 content

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

this_dog = Dog("Buddy", 3)

# inheritance example
class GuideDog(Dog): # note brackets
    job = "Guide the owner"       # Class attribute (shared by all guide dogs)

    def __init__(self, name, age, trained=False): # note default value
        super().__init__(name, age) # note super()
        self.trained = trained # instance attribute; also attribute unique to this subclass

    def guide(self): # new method
        if self.trained:
            print(f"{self.name} is guiding their owner.")
        else:
            print(f"{self.name} is not trained to guide.")

    def bark(self): # overriding method
        print(f"{self.name} says: I'm a guide dog. Woof!")

# class vs instance attributes
dog1 = GuideDog("Rex", 5)
dog2 = GuideDog("Rover", 3)

print(dog1.job)         # Same class attribute: Guide the owner
print(dog2.job)         # Same class attribute: Guide the owner

## Modify class attribute (will affect all instances)
GuideDog.job = "Helping the visually impaired"
print(dog1.job)         # Updated class attribute: Helping the visually impaired
print(dog2.job)         # Same update reflected here

## Overriding class attribute on one instance
dog2.job = "Emotional support"  # This creates an instance attribute named job
print(dog2.job)                 # Now uses instance attribute: Emotional support
print(dog1.job)                 # Still uses class attribute: Helping the visually impaired

## Modifying one instance attribute does not modify other instances
print(dog1.trained)
print(dog2.trained)
dog1.trained = True
print(dog1.trained)
print(dog2.trained)

# global variables, again
dog_count = 0
all_dogs = []

def register_dog(dog):
    global dog_count
    dog_count += 1 # without global, this would give error
    all_dogs.append(dog) # don't need global because not reassigning a variable
    print(f"{dog.name} has been registered. Total dogs: {dog_count}")

my_guide_dog = GuideDog("Max", 5, True)
your_guide_dog = GuideDog("Bella", 2, False)
register_dog(this_dog)
register_dog(my_guide_dog)
register_dog(your_guide_dog)

for dog in all_dogs:
    dog.describe()
    dog.bark()
    if isinstance(dog, GuideDog):
        dog.guide()  # Call guide method only for GuideDog instances
    print()  # Print a newline for better readability
