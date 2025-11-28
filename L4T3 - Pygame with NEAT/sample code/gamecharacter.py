class GameCharacter:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.health = 100
        self.alive = True

    def take_damage(self, amount):
        self.health = max(self.health - amount, 0)
        if self.health <= 0:
            self.alive = False
        print(f"{self.name} took {amount} damage! Current health: {self.health}. Alive: {self.alive}.")

    def level_up(self):
        self.level += 1
        print(f"Level up! {self.name} is now level {self.level}!")

    def heal(self, amount):
        if self.alive:
            self.health = min(self.health + amount, 100)
            print(f"Healed. {self.name} now has health {self.health}.")
        else:
            print(f"Can't heal. {self.name} is no longer alive.")

    def describe(self):
        print(f"{self.name} is level {self.level} and has health {self.health}. Alive: {self.alive}.")


my_character = GameCharacter("Hiccup", 1)
my_character.describe()
my_character.take_damage(30)
my_character.level_up()
my_character.heal(20)
my_character.take_damage(100)
my_character.heal(20)
my_character.describe()