import random
class Card():
    def __init__(self, suit, value): # constructor method
        self.suit = suit      # attribute - property / data
        self.value = value

    def show(self):           # another method
        print(self.value, "of", self.suit)

suits = ["clubs", "hearts", "spades", "diamonds"]
my_card = Card(random.choice(suits), random.randint(1, 13))
print(my_card.suit)           # accessing an attribute - no brackets
my_card.show()                # using a method - with brackets

print("4 of diamonds to win!!!")
num_cards = 0
while True:
    card = Card(random.choice(suits), random.randint(1, 13))
    card.show()
    num_cards += 1
    if card.suit == "diamonds" and card.value == 4:
        print(f"you won in {num_cards} cards")
        break