class Cat:
    def __init__(self, name): # init method
        self.name = name # attribute
        self.age = 1

    def meow(self): # method
        print(f"{self.name} says meow!")

    def birthday(self):
        self.age = self.age + 1
        print(f"{self.name} is now {self.age} years old.")

class DomesticCat(Cat): # subclass of Cat
    def __init__(self, name):
        super().__init__(name)
        self.toy = "laser pointer"
        self.breed = "calico"

# create a subclass called WildCat
class WildCat(Cat): # subclass of Cat
    def __init__(self, name):
        super().__init__(name)
        self.toy = "fish"

    def slap(self):
        print(f"{self.name} got slapped by the fish, meow!!")

my_cat = Cat("Mr Meow") # object
print(my_cat.name)
print(my_cat.age)
my_cat.meow()
my_cat.birthday()

your_cat = DomesticCat("Ginger")
print(your_cat.age)
your_cat.birthday()
print(your_cat.breed)
# print(my_cat.breed)

wild_cat = WildCat("Lion")
wild_cat.slap()