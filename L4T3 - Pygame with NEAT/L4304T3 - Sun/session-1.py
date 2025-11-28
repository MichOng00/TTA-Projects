class Dog:
    job = "basketball player"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def play_fetch(self):
        print(f"{self.name} went to fetch!")

dog = Dog("Rover", 2)
print(dog.age)
dog.play_fetch()

other_dog = Dog("Bud", 3)
print(other_dog.age)