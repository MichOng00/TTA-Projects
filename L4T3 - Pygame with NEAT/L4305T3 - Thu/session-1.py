# introduction to classes
class Dog:
    # init method
    def __init__(self, name, age):
        self.name = name # attribute - store data
        self.age = age
        self.sitting = False # flag attribute
        self.energy = 100

    def bark(self): # method
        print(f"{self.name} says Woof!")

    def sit(self):
        self.sitting = True
        print(f"{self.name} is sitting.")

    def stand(self):
        self.sitting = False
        print(f"{self.name} is standing.")

    def happy_birthday(self):
        self.age += 1

    def move(self):
        self.energy -= 20
        if self.energy <= 0:
            print(f"{self.name} needs to sleep")

    def sleep(self):
        self.energy = 100

    # write a method that describes all of the dog's attributes
    def desc(self):
        print(f"{self.name} is {self.age} years old. Sitting: {self.sitting}")

class GuideDog(Dog): # subclass of Dog class
    def __init__(self, name, age):
        super().__init__(name, age) # inherit init method from Dog class
        self.learn = False

    def train(self):
        print("Training")
        # time.sleep(5)
        self.learn = True

my_dog = Dog("Max", 2) # object
print(type(my_dog))
print(my_dog.name)
my_dog.bark()

new_dog = Dog("Rover", 1)
new_dog.bark()
new_dog.happy_birthday()
print(new_dog.age)

new_dog.desc()