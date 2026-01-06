import random
import json

# Read categories from file
with open('categories.txt', 'r') as f:
    categories = [line.strip() for line in f.readlines()]

# Randomly select 20 categories
# selected_categories = random.sample(categories, 345)

# Print the list with double quotes
print(json.dumps(categories))
